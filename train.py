import warnings ; warnings.simplefilter(action='ignore', category=FutureWarning)

import os
from time import time
from itertools import chain
from argparse import ArgumentParser, Namespace

import torch
import torch.nn.functional as F
from torch.utils.data import DistributedSampler, DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim import AdamW
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from tensorboardX import SummaryWriter

from data import MelDataset, mel_spectrogram, get_dataset_filelist
from audio import FFTParams
from models import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_loss, discriminator_loss
from utils import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint, copy_config, load_config, seed_everything

assert torch.cuda.is_available(), '>> you must have GPU with cuda to train!!'
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def train(rank:int, a:Namespace, h:Namespace):
    ''' Init '''
    if h.num_gpus > 1:
        init_process_group(
            backend=h.dist_config['dist_backend'],
            init_method=h.dist_config['dist_url'],
            world_size=h.dist_config['world_size'] * h.num_gpus,
            rank=rank,
        )

    torch.cuda.manual_seed(h.seed)
    device = torch.device(f'cuda:{rank:d}')

    ''' Log '''
    if rank == 0:
        print('>> log_path:', a.log_path)
        os.makedirs(a.log_path, exist_ok=True)
        sw = SummaryWriter(os.path.join(a.log_path, 'logs'))

    ''' Model '''
    generator = Generator(h).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    if rank == 0:
        print(generator)
        print('>> param_cnt:', sum([p.numel() for p in generator.parameters() if p.requires_grad]))

    ''' Ckpt '''
    state_dict_g  = None
    state_dict_do = None
    steps = 0
    last_epoch = -1

    ckpt_dp = None
    if   os.path.isdir(a.load):     ckpt_dp = a.load
    elif os.path.isdir(a.log_path): ckpt_dp = a.log_path
    if ckpt_dp:
        print(f'>> try resume from ckpt folder {ckpt_dp}')
        cp_g = scan_checkpoint(ckpt_dp, 'g_')
        if cp_g:
            print('>> found generator ckpt:', cp_g)
            state_dict_g = load_checkpoint(cp_g,  device)

        cp_do = scan_checkpoint(ckpt_dp, 'do_')
        if cp_do:
            print('>> found discriminator & optimizer ckpt:', cp_do)
            state_dict_do = load_checkpoint(cp_do, device)

        generator.load_state_dict(state_dict_g['generator'])
        mpd.load_state_dict(state_dict_do['mpd'])
        msd.load_state_dict(state_dict_do['msd'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']

    if h.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        mpd = DistributedDataParallel(mpd, device_ids=[rank]).to(device)
        msd = DistributedDataParallel(msd, device_ids=[rank]).to(device)

    ''' Optimizer & Scheduler '''
    optim_g = AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_d = AdamW(chain(msd.parameters(), mpd.parameters()), h.learning_rate, betas=[h.adam_b1, h.adam_b2])

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    sched_g = ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    sched_d = ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    ''' Data '''
    train_filelist, valid_filelist = get_dataset_filelist(a)
    fft_param = FFTParams(sr=h.sampling_rate, n_fft=h.n_fft, n_mel=h.num_mels, hop_size=h.hop_size, win_size=h.win_size, fmin=h.fmin, fmax=h.fmax)
    fft_param_for_loss = FFTParams(sr=h.sampling_rate, n_fft=h.n_fft, n_mel=h.num_mels, hop_size=h.hop_size, win_size=h.win_size, fmin=h.fmin, fmax=h.fmax_for_loss)
    trainset = MelDataset(train_filelist, h.segment_size, fft_param, fft_param_for_loss, split=True, shuffle=False if h.num_gpus > 1 else True, device=device)
    sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None
    trainloader = DataLoader(trainset, batch_size=h.batch_size, shuffle=False, sampler=sampler, num_workers=h.num_workers, pin_memory=True, drop_last=True)
    if rank == 0:
        validset = MelDataset(valid_filelist, h.segment_size, fft_param, fft_param_for_loss, split=False, shuffle=False, device=device)
        validloader = DataLoader(validset, batch_size=1, shuffle=False, sampler=None, num_workers=1, pin_memory=True, drop_last=False)
    melspec_full = lambda y: mel_spectrogram(y, fft_param_for_loss)

    ''' Train '''
    generator.train()
    mpd.train()
    msd.train()
    for epoch in range(max(0, last_epoch), a.epochs):
        if rank == 0:
            ts_epoch = time()
            print(f'[Epoch {epoch + 1}]')

        if h.num_gpus > 1:
            sampler.set_epoch(epoch)

        for x, y, y_mel in trainloader:
            if rank == 0: ts_batch = time()

            x     = x    .to(device, non_blocking=True)     # [B=16, M=80, L=32], mel_input (low-band melspec)
            y_mel = y_mel.to(device, non_blocking=True)     # [B=16, M=80, L=32], mel_tagret (full-band melspec)
            y     = y    .to(device, non_blocking=True).unsqueeze(1)    # [B=16, C=1, T=8192]

            y_hat = generator(x)    # [B=16, C=1, T=8192]
            y_hat_mel = melspec_full(y_hat.squeeze(1))   # [B=16, M=80, L=32]

            # Discriminator
            optim_d.zero_grad()
            y_g_hat_detach = y_hat.detach()
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat_detach)
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat_detach)
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)
            loss_disc_all = loss_disc_s + loss_disc_f
            loss_disc_all.backward()
            optim_d.step()

            # Generator
            optim_g.zero_grad()
            loss_mel = F.l1_loss(y_mel, y_hat_mel)
            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_hat)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
            loss_gen_all = loss_gen_s + loss_gen_f + (loss_fm_s + loss_fm_f) + loss_mel * 45
            loss_gen_all.backward()
            optim_g.step()

            steps += 1

            if rank == 0:
                if steps % a.stdout_interval == 0:
                    print(f'>> [Steps {steps:d}] loss_gen: {loss_gen_all.item():4.3f}, loss_mel: {loss_mel.item():4.3f} ({time() - ts_batch:4.3f}s/b)')

                if steps % a.summary_interval == 0:
                    sw.add_scalar('train/loss_gen', loss_gen_all.item(), steps)
                    sw.add_scalar('train/loss_mel', loss_mel    .item(), steps)

                if steps % a.save_interval == 0 and steps != 0:
                    fp_gen = f'{a.log_path}/g_{steps:08d}'
                    ckpt_gen = {
                        'generator': (generator.module if h.num_gpus > 1 else generator).state_dict(),
                    }
                    save_checkpoint(fp_gen, ckpt_gen)
                    fp_disc = f'{a.log_path}/do_{steps:08d}'
                    ckpt_disc = {
                        'mpd': (mpd.module if h.num_gpus > 1 else mpd).state_dict(),
                        'msd': (msd.module if h.num_gpus > 1 else msd).state_dict(),
                        'optim_g': optim_g.state_dict(),
                        'optim_d': optim_d.state_dict(),
                        'steps': steps,
                        'epoch': epoch,
                    }
                    save_checkpoint(fp_disc, ckpt_disc)

                if steps % a.eval_interval == 0:
                    generator.eval()
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

                    loss_mel_val = 0.0
                    with torch.no_grad():
                        for j, (x, y, y_mel) in enumerate(validloader):
                            x     = x    .to(device, non_blocking=True)   # [B=1, M=80, L=32]
                            y_mel = y_mel.to(device, non_blocking=True)   # [B=1, M=80, L=32]

                            y_hat = generator(x)    # [B=1, C=1, T=8192]
                            y_hat_mel = melspec_full(y_hat.squeeze(1))   # [B=1, M=80, L=32]
                            loss_mel_val += F.l1_loss(y_mel, y_hat_mel).item()

                            if j <= 4:
                                if steps == 0:
                                    sw.add_audio(f'gt/y_{j}', y[0], steps, h.sampling_rate)
                                    sw.add_figure(f'gt/y_spec_{j}', plot_spectrogram(y_mel[0].cpu().numpy()), steps)

                                sw.add_audio(f'gen/y_hat_{j}', y_hat[0][0], steps, h.sampling_rate)
                                sw.add_figure(f'gen/y_hat_spec_{j}', plot_spectrogram(y_hat_mel[0].cpu().numpy()), steps)

                        sw.add_scalar('valid/loss_mel', loss_mel_val / len(validloader.dataset), steps)

                    generator.train()

        sched_g.step()
        sched_d.step()

        if rank == 0:
            print(f'>> [Epoch {epoch + 1}] time cost {time() - ts_epoch:.5f}s\n')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--config',     default='configs/config_v3.json')
    parser.add_argument('--input_wavs_dir',   default='data/LJSpeech-1.1/wavs')
    parser.add_argument('--input_train_file', default='filelist/LJSpeech-1.1/training.txt')
    parser.add_argument('--input_valid_file', default='filelist/LJSpeech-1.1/validation.txt')
    parser.add_argument('-E', '--epochs',     default=3100, type=int)
    parser.add_argument('--stdout_interval',  default=5,    type=int)
    parser.add_argument('--summary_interval', default=100,  type=int)
    parser.add_argument('--eval_interval',    default=1000, type=int)
    parser.add_argument('--save_interval',    default=5000, type=int)
    parser.add_argument('--log_path', default='log', help='save log folder')
    parser.add_argument('--load', help='resume from pretrained ckpt folder')
    a = parser.parse_args()

    h = load_config(a.config)
    copy_config(a.config, os.path.join(a.log_path, 'config.json'))

    seed_everything(h.seed)
    if torch.cuda.is_available():
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = h.batch_size // h.num_gpus
        print('>> batch_size per GPU:', h.batch_size)

    if h.num_gpus > 1:
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h))
    else:
        train(0, a, h)
