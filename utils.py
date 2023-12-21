import os
import glob
import random
import shutil
from pathlib import Path
from typing import *

import torch
import numpy as np
import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

device = 'cuda' if torch.cuda.is_available() else torch.device('cpu')

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


def seed_everything(seed:int=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def copy_config(config, config_name, path):
    t_path = os.path.join(path, config_name)
    if config != t_path:
        os.makedirs(path, exist_ok=True)
        shutil.copyfile(config, os.path.join(path, config_name))


def plot_spectrogram(spectrogram) -> Figure:
    plt.clf()
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect='auto', origin='lower', interpolation='none')
    plt.colorbar(im, ax=ax)
    fig.canvas.draw()
    return fig


def load_checkpoint(fp:Path, device:str='cpu'):
    assert os.path.isfile(fp)
    print(f'>> load ckpt from {fp}')
    return torch.load(fp, map_location=device)


def save_checkpoint(fp:Path, ckpt:Any):
    print(f'>> save ckpt to {fp}')
    torch.save(ckpt, fp)


def scan_checkpoint(cp_dir:str, prefix:str) -> str:
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0: return None
    return sorted(cp_list)[-1]
