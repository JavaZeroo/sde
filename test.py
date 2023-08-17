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

from rich import print
import time

from utils.Models import MLP
from utils.Datasets import BBdataset, MNISTdataset
from utils.utils import plot_source_and_target_mnist, binary, save_gif_frame_mnist
from utils.data_utils import gen_mnist_data, reverse_normalize_dataset, normalize_dataset_with_metadata
import argparse

parser = argparse.ArgumentParser(description='Train SDE')
parser.add_argument('--seed', type=int, default=233)
parser.add_argument('--task', type=str, default='gaussian2minst')
parser.add_argument('--change_epsilons', action='store_true')
parser.add_argument('--checkpoint', type=str, default=None)