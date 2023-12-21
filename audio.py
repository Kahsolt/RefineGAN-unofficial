from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from numpy import ndarray
from numpy.typing import NDArray
from scipy.io.wavfile import read as read_wav, write as write_wav
from librosa.util import normalize
from librosa.filters import mel as mel_filter
from librosa.core import resample, load

MAX_WAV_VALUE = 32768.0

Wav = NDArray[np.float32]   # vrng norm to [-1, 1]


def load_wav(fp:Path, sr:int=None) -> Wav:
    sr_orig, wav = read_wav(fp)
    wav = wav / MAX_WAV_VALUE
    wav = wav.astype(np.float32)
    if sr and sr != sr_orig:
        wav = resample(wav, sr_orig, sr, res_type='kaiser_best', fix=True, scale=True)
    return wav


def save_wav(fp:Path, wav:Wav, sr:int):
    if wav.min() < -1 or wav.max() > 1:
        print(f'>> warn: clip wav vrng [{wav.min()} ,{wav.max()}] => [-1, 1]')
        wav = wav.clip(-1.0, 1.0)
    wav = wav * MAX_WAV_VALUE
    wav = wav.astype(np.int16)
    write_wav(fp, sr, wav)


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


mel_basis = {}
hann_window = {}

def mel_spectrogram(y:Tensor, n_fft:int, n_mels:int, sampling_rate:int, hop_size:int, win_size:int, fmin:int, fmax:int) -> Tensor:
    if torch.min(y) < -1.0: print('min value is:', torch.min(y))
    if torch.max(y) > +1.0: print('max value is:', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = mel_filter(sr=sampling_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
        mel_basis[f'{fmax}_{y.device}'] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)
    window = hann_window[str(y.device)]
    basis = mel_basis[f'{fmax}_{y.device}']

    P = (n_fft - hop_size) // 2
    y = y.unsqueeze(1)
    y = F.pad(y, (P, P), mode='reflect')
    y = y.squeeze(dim=1)

    spec = torch.stft(y, n_fft, hop_size, win_size, window, onesided=True, return_complex=True)
    spec = torch.abs(spec)  # modulo (magintude)
    spec = torch.matmul(basis, spec)
    spec = dynamic_range_compression_torch(spec)
    return spec
