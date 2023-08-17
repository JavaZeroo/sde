import numpy as np
import matplotlib.pyplot as plt
import pickle

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, StepLR, OneCycleLR
from pathlib import Path
from sklearn.datasets import *
import segmentation_models_pytorch as smp

from rich.spinner import Spinner
from rich.panel import Panel
from rich.pretty import Pretty
from rich.console import Console
from rich import print
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
import time

from utils.Models import MLP
from utils.Datasets import BBdataset, MNISTdataset
from utils.utils import plot_source_and_target_mnist, binary, save_gif_frame_mnist
from utils.data_utils import gen_mnist_data, reverse_normalize_dataset, normalize_dataset_with_metadata
import argparse

import logging
from rich.logging import RichHandler


def main():
    parser = argparse.ArgumentParser(description='Train SDE')
    parser.add_argument('--seed', type=int, default=233)
    parser.add_argument('--task', type=str, default='gaussian2minst')
    parser.add_argument('--change_epsilons', action='store_true')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--scheduler', type=str, default=None)

    args = parser.parse_args()

    seed = args.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    experiment_name = args.task  # 你可以根据需要动态设置这个变量

    log_dir = Path('experiments') / experiment_name / time.strftime("%Y-%m-%d/%H_%M_%S/")
    log_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir = log_dir
    
    main_worker(args)


def main_worker(args):
    
    console = Console(record=True, color_system='truecolor')
    pretty = Pretty(args.__dict__, expand_all=True)
    panel = Panel(pretty, title='Arguments', expand=False, highlight=True)
    console.log(panel)
    console.rule('Training')
    console.save_text('test.log')
    
    pass

if __name__ == '__main__':
    main()
    pass