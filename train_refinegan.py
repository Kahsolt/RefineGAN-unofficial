from models import Refiner, MultiResolutionDiscriminator, multi_param_melspec_loss, envelope_loss, refinegan_generator_loss, refinegan_discriminator_loss
from train import *


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
    generator = Refiner(h).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    mrd = MultiResolutionDiscriminator().to(device)

    if rank == 0:
        print(generator)
        print('>> param_cnt:', sum([p.numel() for p in generator.parameters() if p.requires_grad]))

    ''' Ckpt '''
    state_dict_g  = None
    state_dict_do = None
    steps = 0
    last_epoch = -1

    ckpt_dp = None
    if   a.load     and os.path.isdir(a.load):     ckpt_dp = a.load
    elif a.log_path and os.path.isdir(a.log_path): ckpt_dp = a.log_path
    if ckpt_dp:
        print(f'>> try resume from ckpt folder {ckpt_dp}')
        cp_g = scan_checkpoint(ckpt_dp, 'g_')
        if cp_g:
            print('>> found generator ckpt:', cp_g)
            state_dict_g = load_checkpoint(cp_g,  device)
            generator.load_state_dict(state_dict_g['generator'])

        cp_do = scan_checkpoint(ckpt_dp, 'do_')
        if cp_do:
            print('>> found discriminator & optimizer ckpt:', cp_do)
            state_dict_do = load_checkpoint(cp_do, device)
            mpd.load_state_dict(state_dict_do['mpd'])
            mrd.load_state_dict(state_dict_do['mrd'])
            steps = state_dict_do['steps'] + 1
            last_epoch = state_dict_do['epoch']

    if h.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        mpd = DistributedDataParallel(mpd, device_ids=[rank]).to(device)
        mrd = DistributedDataParallel(mrd, device_ids=[rank]).to(device)

    ''' Optimizer & Scheduler '''
    optim_g = AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_d = AdamW(chain(mrd.parameters(), mpd.parameters()), h.learning_rate, betas=[h.adam_b1, h.adam_b2])

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    sched_g = ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    sched_d = ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    ''' Data '''
    train_filelist, valid_filelist = get_dataset_filelist(a)
    fft_param = FFTParams(sr=h.sampling_rate, n_fft=h.n_fft, n_mel=h.num_mels, hop_size=h.hop_size, win_size=h.win_size, fmin=h.fmin, fmax=h.fmax)
    fft_param_for_loss = FFTParams(sr=h.sampling_rate, n_fft=h.n_fft, n_mel=h.num_mels, hop_size=h.hop_size, win_size=h.win_size, fmin=h.fmin, fmax=h.fmax_for_loss)
    trainset = MelDataset(train_filelist, h.segment_size, fft_param, fft_param_for_loss, split=True, wav_tmpl=True, shuffle=False if h.num_gpus > 1 else True, device=device)
    sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None
    trainloader = DataLoader(trainset, batch_size=h.batch_size, shuffle=False, sampler=sampler, num_workers=h.num_workers, pin_memory=True, drop_last=True)
    if rank == 0:
        validset = MelDataset(valid_filelist, h.segment_size, fft_param, fft_param_for_loss, split=False, wav_tmpl=True, shuffle=False, device=device)
        validloader = DataLoader(validset, batch_size=1, shuffle=False, sampler=None, num_workers=1, pin_memory=True, drop_last=False)
    melspec_full = lambda y: mel_spectrogram(y, fft_param_for_loss)

    ''' Train '''
    generator.train()
    mpd.train()
    mrd.train()
    for epoch in range(max(0, last_epoch), a.epochs):
        if rank == 0:
            ts_epoch = time()
            print(f'[Epoch {epoch + 1}]')

        if h.num_gpus > 1:
            sampler.set_epoch(epoch)

        for x, y, y_tmpl, y_mel in trainloader:
            if rank == 0: ts_batch = time()

            x      = x     .to(device, non_blocking=True)               # [B=16, M=80, L=32], mel_input (low-band melspec)
            y_tmpl = y_tmpl.to(device, non_blocking=True).unsqueeze(1)  # [B=16, C=1, T=8192]
            y_mel  = y_mel .to(device, non_blocking=True)               # [B=16, M=80, L=32], mel_tagret (full-band melspec)
            y      = y     .to(device, non_blocking=True).unsqueeze(1)  # [B=16, C=1, T=8192]

            y_hat = generator(x, y_tmpl)    # [B=16, C=1, T=8192]
            y_hat_mel = melspec_full(y_hat.squeeze(1))   # [B=16, M=80, L=32]

            # Discriminator
            optim_d.zero_grad()
            y_hat_detach = y_hat.detach()
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_hat_detach)
            loss_disc_f = refinegan_discriminator_loss(y_df_hat_r, y_df_hat_g)
            y_dr_hat_r, y_dr_hat_g = mrd(y, y_hat_detach)
            loss_disc_r = refinegan_discriminator_loss(y_dr_hat_r, y_dr_hat_g)
            loss_disc_all = loss_disc_f + loss_disc_r
            loss_disc_all.backward()
            optim_d.step()

            # Generator
            optim_g.zero_grad()
            loss_mstft = multi_param_melspec_loss(y.squeeze(1), y_hat.squeeze(1), mrd.resolutions, fft_param_for_loss)
            loss_env = envelope_loss(y, y_hat)
            _, y_df_hat_g, _, _ = mpd(y, y_hat)
            _, y_dr_hat_g = mrd(y, y_hat)
            loss_gen_f = refinegan_generator_loss(y_df_hat_g)
            loss_gen_r = refinegan_generator_loss(y_dr_hat_g)
            loss_gen_all = (loss_gen_f + loss_gen_r) + loss_env + loss_mstft * 45   # NOTE: how much is λ?
            loss_gen_all.backward()
            optim_g.step()

            steps += 1

            if rank == 0:
                if steps % a.stdout_interval == 0:
                    print(f'>> [Steps {steps:d}] loss_gen: {loss_gen_all.item():4.3f}, loss_mstft: {loss_mstft.item():4.3f} ({time() - ts_batch:4.3f}s/b)')

                if steps % a.summary_interval == 0:
                    sw.add_scalar('train/loss_gen', loss_gen_all.item(), steps)
                    sw.add_scalar('train/loss_mstft', loss_mstft.item(), steps)

                if steps % a.save_interval == 0 and steps != 0:
                    fp_gen = f'{a.log_path}/g_{steps:08d}'
                    ckpt_gen = {
                        'generator': (generator.module if h.num_gpus > 1 else generator).state_dict(),
                    }
                    save_checkpoint(fp_gen, ckpt_gen)
                    fp_disc = f'{a.log_path}/do_{steps:08d}'
                    ckpt_disc = {
                        'mpd': (mpd.module if h.num_gpus > 1 else mpd).state_dict(),
                        'mrd': (mrd.module if h.num_gpus > 1 else mrd).state_dict(),
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
                        for j, (x, y, y_tmpl, y_mel) in enumerate(validloader):
                            x      = x     .to(device, non_blocking=True)               # [B=1, M=80, L=32]
                            y_tmpl = y_tmpl.to(device, non_blocking=True).unsqueeze(1)   # [B=1, C=1, T=8192]
                            y_mel  = y_mel .to(device, non_blocking=True)               # [B=1, M=80, L=32]

                            y_hat = generator(x, y_tmpl)    # [B=1, C=1, T=8192]
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
    parser.add_argument('-c', '--config',     default='configs/refinegan.json')
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
