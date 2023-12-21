from pathlib import Path
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from numpy import ndarray
from numpy.typing import NDArray
from scipy.io.wavfile import write as write_wav
from librosa.core import load
from librosa.util import normalize
from librosa.filters import mel as mel_filter

Wav = NDArray[np.float32]   # vrng norm to [-1, 1]


@dataclass
class FFTParams:
    sr: int
    n_fft: int
    n_mel: int
    hop_size: int
    win_size: int
    fmin: int
    fmax: int


def load_wav(fp:Path, sr:int=None) -> Wav:
    wav, sr = load(fp, sr=sr, res_type='kaiser_best', dtype=np.float32)
    return wav


def save_wav(fp:Path, wav:Wav, sr:int):
    if wav.min() < -1 or wav.max() > 1:
        print(f'>> warn: clip wav vrng [{wav.min()} ,{wav.max()}] => [-1, 1]')
        wav = wav.clip(-1.0, 1.0)
    write_wav(fp, sr, wav)


def dynamic_range_compression(x:ndarray, C:float=1, eps:float=1e-5) -> ndarray:
    return np.log(np.clip(x, a_min=eps, a_max=None) * C)


def dynamic_range_decompression(x:ndarray, C:float=1) -> ndarray:
    return np.exp(x) / C


def dynamic_range_compression_torch(x:Tensor, C:float=1, eps:float=1e-5) -> Tensor:
    return torch.log(torch.clamp(x, min=eps) * C)


def dynamic_range_decompression_torch(x:Tensor, C:float=1) -> Tensor:
    return torch.exp(x) / C


mel_basis: Dict[str, Tensor] = {}
window_func: Dict[str, Tensor] = {}

def mel_spectrogram(y:Tensor, h:FFTParams) -> Tensor:
    assert len(y.shape), f'>> y.shape should be like [B, T], but got {tuple(y.shape)}'
    if y.min() < -1.0 or y.max() > 1.0:
        print(f'warn: wav vrng is [{y.min()}, {y.max()}]')

    global mel_basis, window_func
    win_fn_name = str(y.device)
    mel_fn_name = f'{y.device}_{h.fmax}'
    if mel_fn_name not in mel_basis:
        mel = mel_filter(sr=h.sr, n_fft=h.n_fft, n_mels=h.n_mel, fmin=h.fmin, fmax=h.fmax)
        mel_basis[mel_fn_name] = torch.from_numpy(mel).float().to(y.device)
        window_func[win_fn_name] = torch.hann_window(h.win_size).to(y.device)
    window = window_func[win_fn_name]
    basis = mel_basis[mel_fn_name]

    '''
    # https://pytorch.org/docs/stable/generated/torch.stft.html
    torch.stft is wired about input/output length when center=True:
        len_out = 1 + len_in // hop_length
    hence we have:
        wav: [B=1, T=128] => spec(hop_length=128): [B=1, F, L=2]
        wav: [B=1, T=127] => spec(hop_length=128): [B=1, M, L=1]
    '''
    wavlen = y.shape[-1]
    if wavlen % h.hop_size == 0:
        y = y[:, 1:]       # remove one sample to align with k*hop_size-1
    else:
        pass
        # safe to ignore
        # n_seg = int(np.ceil(wavlen / h.hop_size))
        # tgtlen = n_seg * h.hop_size - 1
        # difflen = tgtlen - wavlen
        # y = y.unsqueeze(dim=1)      # [B, C=1, T]
        # y = F.pad(y, (difflen//2, difflen-difflen//2), mode='reflect')
        # y = y.squeeze(dim=1)        # [B, T]

    spec = torch.stft(y, h.n_fft, h.hop_size, h.win_size, window, center=True, onesided=True, return_complex=True)
    mag = torch.abs(spec)   # modulo
    mel = torch.matmul(basis, mag)
    mel = dynamic_range_compression_torch(mel)
    return mel
