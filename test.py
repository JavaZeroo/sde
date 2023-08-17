import numpy as np
import matplotlib.pyplot as plt
import pickle

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, StepLR, OneCycleLR
from pathlib import Path
from sklearn.datasets import *

from rich.panel import Panel
from rich.pretty import Pretty
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
import time

from utils.Datasets import BBdataset, MNISTdataset
from utils.utils import plot_source_and_target_mnist, binary, save_gif_frame_mnist
from utils.data_utils import gen_mnist_data, reverse_normalize_dataset, normalize_dataset_with_metadata, get_ds
from utils.model_utils import get_model_before_after
import argparse



def main():
    parser = argparse.ArgumentParser(description='Train SDE')
    parser.add_argument('--seed', type=int, default=233)
    parser.add_argument('--task', type=str, default='gaussian2minst')
    parser.add_argument('--change_epsilons', action='store_true')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--scheduler', type=str, default=None)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--iter_nums', type=int, default=1)
    parser.add_argument('--epoch_nums', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=8000)
    parser.add_argument('-n','--normalize', action='store_true')

    args = parser.parse_args()

    seed = args.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    experiment_name = args.task
    log_dir = Path('experiments') / experiment_name / time.strftime("%Y-%m-%d/%H_%M_%S/")
    if args.task == 'gaussian2minst':
        args.dim = 1
    else:
        args.dim = 2
        
    log_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir = log_dir
    args.ds_cached_dir = args.log_dir / 'data'

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main_worker(args)

def train(args, model, train_dl, optimizer, scheduler, loss_fn, before_train=None, after_train=None):
    losses = 0
    for training_data in train_dl:
        training_data = training_data[:,0].float().cpu()
        x, y = training_data[:, :-1], training_data[:, -1:]
        if before_train is not None:
            x = before_train(x)
        x = x.to(args.device)
        y = y.to(args.device)
        pred = model(x)
        if after_train is not None:
            pred = after_train(pred)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()
        losses += loss.item() / len(train_dl)
    return losses

def main_worker(args):
    console = Console(record=True, color_system='truecolor')
    pretty = Pretty(args.__dict__, expand_all=True)
    panel = Panel(pretty, title='Arguments', expand=False, highlight=True)
    console.log(panel)
    console.log(f"Saving to {args.log_dir}")

    model, before_train, after_train = get_model_before_after(args)
        
    if args.checkpoint is not None:
        try:
            model.load_state_dict(torch.load(args.checkpoint))
        except:
            console.log(":warning-emoji: [bold red blink] load checkpoint failed [/]\ncheckpoint: {}\nExit".format(args.checkpoint))
            return None
        console.log("load checkpoint from {}".format(args.checkpoint))
        
    console.log(f"Model {model.__class__.__name__} Parameters: {int(sum(p.numel() for p in model.parameters())/1e6)}M")
    
    real_metadata = pickle.loads(open(args.ds_cached_dir / 'real_mean_std.pkl', 'rb').read())

    torch.save(model.state_dict(), args.log_dir / f'model_{model.__class__.__name__}_final.pth')

    test_ts, test_bridge, test_drift, test_source_sample, _ = gen_mnist_data(nums=1000)

    pred_bridge = torch.zeros_like(test_bridge)
    pred_drift = torch.zeros_like(test_drift)

    pred_bridge[0, :] = test_source_sample
    model.eval()
    sigma=1
    with torch.no_grad():
        for i in range(len(test_ts) - 1):
            dt = (test_ts[i+1] - test_ts[i])
            test_source_sample_reshaped = test_source_sample
            test_ts_reshaped = test_ts[i].repeat(test_source_sample.shape[0]).reshape(-1, 1, 1, 1).repeat(1, 1, 28, 28)
            pred_bridge_reshaped = pred_bridge[i]

            ret = normalize_dataset_with_metadata(real_metadata, source=test_source_sample_reshaped, ts=test_ts_reshaped, bridge=pred_bridge_reshaped)
            test_ts_reshaped = ret['ts']
            pred_bridge_reshaped = ret['bridge']
            test_source_sample_reshaped = ret['source']
            
            x = torch.concat([test_source_sample_reshaped, test_ts_reshaped, pred_bridge_reshaped], axis=1)
            
            if before_train is not None:
                x = before_train(x)
            dydt = model(x.to(args.device)).cpu()
            if after_train is not None:
                pred = after_train(pred)            
            pred_drift[i]=dydt
            dydt = reverse_normalize_dataset(real_metadata, bridge=dydt)['bridge']

            diffusion = sigma * torch.sqrt(dt) * torch.randn_like(dydt)

            pred_bridge[i+1] = pred_bridge[i] + dydt * dt + diffusion[:]

    plot_source_and_target_mnist(test_bridge[-1, :25], binary(pred_bridge[-1, :25]))

    save_gif_frame_mnist(pred_bridge[:, :25], args.log_dir, 'pred.gif')


if __name__ == '__main__':
    main()
    pass