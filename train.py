import warnings ; warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import json
from itertools import chain
from time import time
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
from models import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_loss, discriminator_loss
from utils import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint, copy_config, seed_everything


def train(rank, a, h):
    if h.num_gpus > 1:
        init_process_group(
            backend=h.dist_config['dist_backend'],
            init_method=h.dist_config['dist_url'],
            world_size=h.dist_config['world_size'] * h.num_gpus,
            rank=rank,
        )

    torch.cuda.manual_seed(h.seed)
    device = torch.device(f'cuda:{rank:d}')

    generator = Generator(h).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    if rank == 0:
        print(generator)
        os.makedirs(a.log_path, exist_ok=True)
        print('>> log_path:', a.log_path)

    if os.path.isdir(a.log_path):
        cp_g = scan_checkpoint(a.log_path, 'g_')
        cp_do = scan_checkpoint(a.log_path, 'do_')

    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
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

    optim_g = AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_d = AdamW(chain(msd.parameters(), mpd.parameters()), h.learning_rate, betas=[h.adam_b1, h.adam_b2])

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    scheduler_g = ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    training_filelist, validation_filelist = get_dataset_filelist(a)

    trainset = MelDataset(
        training_filelist,
        h.segment_size,
        h.n_fft,
        h.num_mels,
        h.hop_size,
        h.win_size,
        h.sampling_rate,
        h.fmin,
        h.fmax,
        shuffle=False if h.num_gpus > 1 else True,
        fmax_loss=h.fmax_for_loss,
        device=device,
        finetune=a.finetune,
        base_mels_path=a.input_mels_dir
    )
    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None
    train_loader = DataLoader(
        trainset,
        num_workers=h.num_workers,
        shuffle=False,
        sampler=train_sampler,
        batch_size=h.batch_size,
        pin_memory=True,
        drop_last=True,
    )

    if rank == 0:
        validset = MelDataset(
            validation_filelist,
            h.segment_size,
            h.n_fft,
            h.num_mels,
            h.hop_size,
            h.win_size,
            h.sampling_rate,
            h.fmin,
            h.fmax,
            False,
            False,
            fmax_loss=h.fmax_for_loss,
            device=device,
            finetune=a.finetune,
            base_mels_path=a.input_mels_dir,
        )
        validation_loader = DataLoader(
            validset,
            num_workers=1,
            shuffle=False,
            sampler=None,
            batch_size=1,
            pin_memory=True,
            drop_last=True,
        )

        sw = SummaryWriter(os.path.join(a.log_path, 'logs'))

    generator.train()
    mpd.train()
    msd.train()
    for epoch in range(max(0, last_epoch), a.epochs):
        if rank == 0:
            ts_epoch = time()
            print(f'[Epoch {epoch + 1}]')

        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        for batch in train_loader:
            if rank == 0: ts_batch = time()

            x, y, _, y_mel = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).unsqueeze(1)
            y_mel = y_mel.to(device, non_blocking=True)

            y_g_hat = generator(x)
            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax_for_loss)

            # Discriminator
            optim_d.zero_grad()
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)
            loss_disc_all = loss_disc_s + loss_disc_f
            loss_disc_all.backward()
            optim_d.step()

            # Generator
            optim_g.zero_grad()
            loss_mel = F.l1_loss(y_mel, y_g_hat_mel)
            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel * 45
            loss_gen_all.backward()
            optim_g.step()

            steps += 1

            if rank == 0:
                if steps % a.stdout_interval == 0:
                    print(f'>> [Steps {steps:d}] loss_gen: {loss_gen_all.item():4.3f}, loss_mel: {loss_mel.item():4.3f} ({time() - ts_batch:4.3f}s/b)')

                if steps % a.summary_interval == 0:
                    sw.add_scalar('train/loss_gen', loss_gen_all.item(), steps)
                    sw.add_scalar('train/loss_mel', loss_mel    .item(), steps)

                if steps % a.checkpoint_interval == 0 and steps != 0:
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

                    loss_mel_val = 0
                    with torch.inference_mode():
                        for j, batch in enumerate(validation_loader):
                            x, y, _, y_mel = batch
                            y_g_hat = generator(x.to(device))
                            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
                            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax_for_loss)
                            loss_mel_val += F.l1_loss(y_mel, y_g_hat_mel).item()

                            if j <= 4:
                                if steps == 0:
                                    sw.add_audio(f'gt/y_{j}', y[0], steps, h.sampling_rate)
                                    sw.add_figure(f'gt/y_spec_{j}', plot_spectrogram(x[0]), steps)

                                sw.add_audio(f'gen/y_hat_{j}', y_g_hat[0], steps, h.sampling_rate)
                                y_hat_spec = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)
                                sw.add_figure(f'gen/y_hat_spec_{j}', plot_spectrogram(y_hat_spec.squeeze(0).cpu().numpy()), steps)

                        sw.add_scalar('valid/loss_mel', loss_mel_val / (j + 1), steps)

                    generator.train()

        scheduler_g.step()
        scheduler_d.step()

        if rank == 0:
            print(f'>> [Epoch {epoch + 1}] time cost {time() - ts_epoch:.5f}s\n')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config',              default='configs/config_v3.json')
    parser.add_argument('--input_wavs_dir',      default='data/LJSpeech-1.1/wavs')
    parser.add_argument('--input_mels_dir',      default='ft_dataset')
    parser.add_argument('--input_train_file',    default='filelist/LJSpeech-1.1/training.txt')
    parser.add_argument('--input_valid_file',    default='filelist/LJSpeech-1.1/validation.txt')
    parser.add_argument('-E', '--epochs',        default=3100, type=int)
    parser.add_argument('--stdout_interval',     default=5,    type=int)
    parser.add_argument('--summary_interval',    default=100,  type=int)
    parser.add_argument('--eval_interval',       default=1000, type=int)
    parser.add_argument('--checkpoint_interval', default=5000, type=int)
    parser.add_argument('--log_path', default='log')
    parser.add_argument('--finetune', action='store_true')
    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()
    cfg = json.loads(data)
    h = Namespace(**cfg)
    copy_config(a.config, 'config.json', a.log_path)

    seed_everything(h.seed)
    if torch.cuda.is_available():
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print('Batch size per GPU :', h.batch_size)

    if h.num_gpus > 1:
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h,))
    else:
        train(0, a, h)
