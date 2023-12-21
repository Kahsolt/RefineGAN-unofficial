import os
import math
import random
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np

from audio import load_wav, normalize, mel_spectrogram


def get_dataset_filelist(a):
    with open(a.input_train_file, 'r', encoding='utf-8') as fh:
        lines = fh.read().strip().split('\n')
        train_fps = [Path(a.input_wavs_dir / (x.split('|')[0] + '.wav')) for x in lines]
    with open(a.input_valid_file, 'r', encoding='utf-8') as fh:
        lines = fh.read().strip().split('\n')
        valid_fps = [Path(a.input_wavs_dir / (x.split('|')[0] + '.wav')) for x in lines]
    return train_fps, valid_fps


class MelDataset(Dataset):

    def __init__(self, fps:List[Path], segment_size, n_fft, num_mels,
                 hop_size, win_size, sampling_rate,  fmin, fmax, split=True, shuffle=True,
                 device=None, fmax_loss=None, finetune=False, base_mels_path=None):
        self.fps = fps
        if shuffle: random.shuffle(self.fps)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.device = device
        self.finetune = finetune
        self.base_mels_path = base_mels_path

    def __len__(self):
        return len(self.fps)

    def __getitem__(self, index):
        fp = self.fps[index]
        audio = load_wav(fp, self.sampling_rate)
        if not self.finetune:
            audio = normalize(audio) * 0.95

        audio = torch.from_numpy(audio).unsqueeze(0)

        if not self.finetune:
            if self.split:
                if audio.size(1) >= self.segment_size:
                    max_audio_start = audio.size(1) - self.segment_size
                    audio_start = random.randint(0, max_audio_start)
                    audio = audio[:, audio_start:audio_start+self.segment_size]
                else:
                    audio = F.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

            mel = mel_spectrogram(audio, self.n_fft, self.num_mels, self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax, center=False)
        else:
            mel = np.load(os.path.join(self.base_mels_path, os.path.splitext(os.path.split(fp)[-1])[0] + '.npy'))
            mel = torch.from_numpy(mel)

            if len(mel.shape) < 3:
                mel = mel.unsqueeze(0)

            if self.split:
                frames_per_seg = math.ceil(self.segment_size / self.hop_size)

                if audio.size(1) >= self.segment_size:
                    mel_start = random.randint(0, mel.size(2) - frames_per_seg - 1)
                    mel = mel[:, :, mel_start:mel_start + frames_per_seg]
                    audio = audio[:, mel_start * self.hop_size:(mel_start + frames_per_seg) * self.hop_size]
                else:
                    mel = F.pad(mel, (0, frames_per_seg - mel.size(2)), 'constant')
                    audio = F.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

        mel_loss = mel_spectrogram(audio, self.n_fft, self.num_mels, self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax_loss, center=False)

        return mel.squeeze(), audio.squeeze(0), fp, mel_loss.squeeze()
