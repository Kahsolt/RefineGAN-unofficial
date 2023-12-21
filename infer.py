import os
import json
from pathlib import Path
from argparse import ArgumentParser, Namespace

import torch
from torch import Tensor

from models import Generator
from audio import mel_spectrogram, load_wav, save_wav
from utils import load_checkpoint, seed_everything, device

h = None


def get_mel(x):
    return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)


@torch.inference_mode()
def infer(a):
    generator = Generator(h).to(device)
    ckpt = load_checkpoint(a.ckpt, device)
    generator.load_state_dict(ckpt['generator'])

    os.makedirs(a.output_dir, exist_ok=True)

    generator.eval()
    generator.remove_weight_norm()
    for fp in Path(a.input_dir).iterdir():
        wav = load_wav(fp, h.sampling_rate)
        wav = torch.from_numpy(wav).to(device)
        x = get_mel(wav.unsqueeze(0))
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
    with open(fp_cfg) as f:
        data = f.read()
    cfg = json.loads(data)
    h = Namespace(**cfg)
    seed_everything(h.seed)

    infer(a)
