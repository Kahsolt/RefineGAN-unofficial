import os
from pathlib import Path
from argparse import ArgumentParser, Namespace

import torch
from torch import Tensor

from models import Generator
from audio import load_wav, save_wav, mel_spectrogram, FFTParams
from utils import load_checkpoint, load_config, seed_everything

device = 'cuda' if torch.cuda.is_available() else 'cpu'


@torch.inference_mode()
def infer(a:Namespace, h:Namespace):
    fft_param = FFTParams(
        sr=h.sampling_rate,
        n_fft=h.n_fft,
        n_mel=h.num_mels,
        hop_size=h.hop_size,
        win_size=h.win_size,
        fmin=h.fmin,
        fmax=h.fmax,
    )

    generator = Generator(h).to(device)
    ckpt = load_checkpoint(a.ckpt, device)
    generator.load_state_dict(ckpt['generator'])

    os.makedirs(a.output_dir, exist_ok=True)

    generator.eval()
    generator.remove_weight_norm()
    for fp in Path(a.input_dir).iterdir():
        wav = load_wav(fp, h.sampling_rate)
        wav = torch.from_numpy(wav).unsqueeze(0).to(device)
        x = mel_spectrogram(wav, fft_param)
        y_g_hat: Tensor = generator(x)
        wav_gen = y_g_hat.squeeze().cpu().numpy()

        fp = os.path.join(a.output_dir, fp.stem + '_generated.wav')
        save_wav(fp, wav_gen, h.sampling_rate)
        print(f'>> save to {fp}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('ckpt', help='path to generator ckpt file')
    parser.add_argument('-I', '--input_dir', default='test')
    parser.add_argument('-O', '--output_dir', default='out')
    a = parser.parse_args()

    fp_cfg = Path(a.ckpt).parent / 'config.json'
    h = load_config(fp_cfg)
    seed_everything(h.seed)

    infer(a, h)
