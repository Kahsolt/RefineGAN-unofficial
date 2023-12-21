import random
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from audio import load_wav, normalize, random_pitch_shift, random_energy_rescale, mel_spectrogram, FFTParams


def get_dataset_filelist(a) -> Tuple[List[Path], List[Path]]:
    with open(a.input_train_file, 'r', encoding='utf-8') as fh:
        lines = fh.read().strip().split('\n')
        train_fps = [Path(a.input_wavs_dir) / (x.split('|')[0] + '.wav') for x in lines]
    with open(a.input_valid_file, 'r', encoding='utf-8') as fh:
        lines = fh.read().strip().split('\n')
        valid_fps = [Path(a.input_wavs_dir) / (x.split('|')[0] + '.wav') for x in lines]
    return train_fps, valid_fps


class MelDataset(Dataset):

    def __init__(self, fps:List[Path], segment_size:int, fft_param:FFTParams, fft_param_for_loss:FFTParams=None, split:bool=True, shuffle:bool=False, device:str=None):
        self.fps = fps
        self.segment_size = segment_size
        self.fft_param = fft_param
        self.fft_param_for_loss = fft_param_for_loss or fft_param
        self.split = split
        self.device = device or 'cpu'

        if shuffle: random.shuffle(self.fps)
        self.cache = [None] * len(self.fps)

    def __len__(self):
        return len(self.fps)

    def __getitem__(self, index):
        if self.cache[index] is None:
            fp = self.fps[index]
            wav = load_wav(fp, self.fft_param.sr)       # [T]
            wav = normalize(wav) * 0.95
            wav = torch.from_numpy(wav).unsqueeze(0)    # [B=1, T]
            self.cache[index] = wav
        wav = self.cache[index]

        if not 'data_aug':
            wav = random_pitch_shift(wav)
            wav = random_energy_rescale(wav)

        if self.split:
            wavlen = wav.size(dim=1)
            if wavlen == self.segment_size:
                pass
            elif wavlen > self.segment_size:
                cp = random.randint(0, wavlen - self.segment_size)
                wav = wav[:, cp:cp+self.segment_size]
            else:
                wav = F.pad(wav, (0, self.segment_size - wav.size(1)), mode='constant')

        mel_lowpass = mel_spectrogram(wav, self.fft_param)
        mel_full = mel_spectrogram(wav, self.fft_param_for_loss)

        # [M=80, L=32], [T=8192], [M=80, L=32]
        return mel_lowpass.squeeze(), wav.squeeze(), mel_full.squeeze()
