import random
from pathlib import Path
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from numpy import ndarray
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from scipy.io.wavfile import write as write_wav
from librosa.core import load, stft
from librosa.util import normalize
from librosa.feature import melspectrogram, rms as rms_energy
from librosa.feature.inverse import mel_to_stft, mel_to_audio, griffinlim
from librosa.effects import pitch_shift
from librosa.filters import mel as mel_filter
import pyworld as pw

if not hasattr(np, 'float'): setattr(np, 'float', np.float32)   # tmp fix for librosa-numpy

Wav = NDArray[np.float32]   # vrng norm to [-1, 1]


@dataclass
class FFTParams:
    sr: int = 44100
    n_fft: int = 2048
    n_mel: int = 128
    hop_size: int = 256
    win_size: int = 2048
    fmin: int = 0
    fmax: int = 22050
    f0min: float = 50.0
    f0max: float = 1100.0


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

    spec = torch.stft(y, h.n_fft, h.hop_size, h.win_size, window, onesided=True, return_complex=True)
    mag = torch.abs(spec)   # modulo
    mel = torch.matmul(basis, mag)
    mel = dynamic_range_compression_torch(mel)
    return mel


''' ↓↓↓ RefineGAN ↓↓↓ '''

def random_pitch_shift(y:ndarray, sr:int=22050, zeta_min:int=-12, zeta_max:int=12) -> ndarray:
    shift = random.choice(range(zeta_min, zeta_max+1))   # ~U[ζmin, ζmax]
    if shift == 0: return y
    return pitch_shift(y, sr=sr, n_steps=shift, bins_per_octave=12, res_type='kaiser_best')

def random_energy_rescale(y:ndarray, r_min:float=0.5, r_max:float=2.0) -> ndarray:
    p = np.abs(y).max()
    # NOTE: the essay's p_min/p_max is probably useless, and shouldn't here let p' = 2^α s.t. α ~ U[-1,1]??!
    p_hat = random.uniform(r_min, r_max)
    return y * p_hat / p


def harveset(y:ndarray, fs:int=22050, hop:int=256, f0_min:float=55.0, f0_max:float=760.0) -> ndarray:
    x = y.astype(np.double)
    f0, t = pw.harvest(x, fs=fs, f0_ceil=f0_max, f0_floor=f0_min, frame_period=hop/fs*1000)
    f0 = pw.stonemask(x, f0, t, fs)
    return f0.astype(np.float32)


PITCH_LOW_CUT = 20.0     # NOTE: lower hearing threshold

def align_f0_c0(f0:ndarray, c0:ndarray) -> tuple[ndarray, ndarray]:
    len_f0, len_c0 = len(f0), len(c0)
    d = len_f0 - len_c0
    if   d > 0: c0 = np.pad(c0, (0,  d), mode='constant', constant_values=0.0)
    elif d < 0: f0 = np.pad(f0, (0, -d), mode='constant', constant_values=0.0)
    return f0, c0

def make_pulses(f0:ndarray, c0:ndarray, hop_size:int):
    from numpy import pi
    pulses = []
    for i in range(len(f0)):
        F = f0[i]
        if F < PITCH_LOW_CUT:
            pulse = np.zeros(hop_size)
        else:
            A = c0[i] * np.sqrt(2)
            T = 1 / F
            t = np.arange(0, T,  T / hop_size)
            pulse = A * np.sin(2 * pi * F * t + pi / 2)
        pulses.append(pulse)
    return np.concatenate(pulses, axis=0)

def make_noises(f0:ndarray, c0:ndarray, hop_size:int):
    noises = []
    for i in range(len(f0)):
        F = f0[i]
        if F >= PITCH_LOW_CUT:
            noise = np.zeros(hop_size)
        else:
            A = c0[i] * np.sqrt(2)
            noise = A * np.random.uniform(-1, 1, size=hop_size)
        noises.append(noise)
    return np.concatenate(noises, axis=0)

def speech_template(pitch:ndarray, energy:ndarray, h:FFTParams) -> ndarray:
    assert len(pitch.shape) == len(energy.shape) == 1, f'>> need 1-D arrays but got: pitch {pitch.shape}, energy {energy.shape}'
    assert len(pitch) == len(energy), f'>> length mismatch pitch({len(pitch)}) != energy ({len(energy)}), manually call align_f0_c0() first!'

    pulses = make_pulses(pitch, energy, h.hop_size)
    noises = make_noises(pitch, energy, h.hop_size)
    template = pulses + noises

    if not 'plot':
        plt.clf()
        plt.subplot(311) ; plt.plot(template) ; plt.title(f'tmpl (len={len(template)})')
        plt.subplot(312) ; plt.plot(pitch)    ; plt.title(f'f0 (len={len(pitch)})')
        plt.subplot(313) ; plt.plot(energy)   ; plt.title(f'c0 (len={len(energy)})')
        plt.show()

    return template

def speech_template_approx_from_wav(y:ndarray, h:FFTParams) -> ndarray:
    f0 = harveset(y, fs=h.sr, hop=h.hop_size, f0_min=h.f0min, f0_max=h.f0max)
    c0 = rms_energy(y=y, frame_length=h.win_size, hop_length=h.hop_size)[0]
    f0, c0 = align_f0_c0(f0, c0)
    return speech_template(f0, c0, h)

def speech_template_approx_from_spec(S:ndarray, h:FFTParams, gl_iter:int=12) -> ndarray:
    y_gl = griffinlim(S, n_iter=gl_iter, hop_length=h.hop_size, win_length=h.win_size, n_fft=h.n_fft, window='hann')
    f0 = harveset(y_gl, fs=h.sr, hop=h.hop_size, f0_min=h.f0min, f0_max=h.f0max)
    f0 = gaussian_filter1d(f0, sigma=1, axis=-1, mode='reflect')
    c0 = rms_energy(S=S, frame_length=h.win_size, hop_length=h.hop_size)[0]
    f0, c0 = align_f0_c0(f0, c0)
    return speech_template(f0, c0, h)

def speech_template_approx_from_melspec(M:ndarray, h:FFTParams, power:float=1.0) -> ndarray:
    S = mel_to_stft(M, sr=h.sr, n_fft=h.n_fft, fmin=h.fmin, fmax=h.fmax, power=power)
    return speech_template_approx_from_spec(S, h)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    h = FFTParams(
        sr=44100,
        n_fft=2048,
        n_mel=128,
        hop_size=256,
        win_size=2048,
        fmin=0,
        fmax=22050,
        f0min=50,
        f0max=1100,
    )
    y = load_wav('test/000001.wav', h.sr)
    S = stft(y, n_fft=h.n_fft, hop_length=h.hop_size, win_length=h.win_size, window='hann')
    M = melspectrogram(S=S, sr=h.sr, n_fft=h.n_fft, hop_length=h.hop_size, win_length=h.win_size, window='hann', power=1.0)
    tmpl_y = speech_template_approx_from_wav(y, h)
    tmpl_S = speech_template_approx_from_spec(S, h)
    tmpl_M = speech_template_approx_from_melspec(M, h)

    plt.clf()
    plt.subplot(411) ; plt.title('y')      ; plt.plot(y)
    plt.subplot(412) ; plt.title('tmpl_y') ; plt.plot(tmpl_y)
    plt.subplot(413) ; plt.title('tmpl_S') ; plt.plot(tmpl_S)
    plt.subplot(414) ; plt.title('tmpl_M') ; plt.plot(tmpl_M)
    plt.tight_layout()
    plt.show()
