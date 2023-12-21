import os
import glob
import json
import random
import shutil
from pathlib import Path
from argparse import Namespace
from typing import *

import torch
import numpy as np
from numpy import ndarray
import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def seed_everything(seed:int=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def copy_config(fp_src:str, fp_dst:str):
    if fp_src == fp_dst: return
    os.makedirs(os.path.dirname(fp_dst), exist_ok=True)
    shutil.copyfile(fp_src, fp_dst)


def load_config(fp:Path) -> Namespace:
    with open(fp, 'r', encoding='utf-8') as fh:
        data = fh.read()
    cfg = json.loads(data)
    return Namespace(**cfg)


def plot_spectrogram(melspec:ndarray) -> Figure:
    plt.clf()
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(melspec, aspect='auto', origin='lower', interpolation='none')
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
