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
from utils.data_utils import gen_ds, read_ds_from_pkl
from utils.model_utils import get_model_before_after
import argparse

def check_model_task(args):
    if args.task.startswith('gaussian2mnist'):
        assert args.model in ['tunet++', 'aunet']
        args.time_expand = False
    else:
        assert args.model in ['mlp', 'unet++', 'unet']
        args.time_expand = True

def main():
    parser = argparse.ArgumentParser(description='Train SDE')
    
    parser.add_argument('--task', type=str, default='gaussian2mnist', required=True)
    parser.add_argument('--model', type=str, default='unet++', required=True)

    parser.add_argument('--seed', type=int, default=233)
    parser.add_argument('--change_epsilons', action='store_true')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--scheduler', type=str, default=None)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--iter_nums', type=int, default=1)
    parser.add_argument('--epoch_nums', type=int, default=3)
    parser.add_argument('-b','--batch_size', type=int, default=8000)
    parser.add_argument('-n','--normalize', action='store_true')
    parser.add_argument('--num_workers', type=int, default=20)
    parser.add_argument('--filter_number', type=int)
    
    parser.add_argument('--debug', action='store_true')
    
    
    args = parser.parse_args()
    check_model_task(args)
    
    seed = args.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    experiment_name = args.task
    if args.change_epsilons:
        experiment_name += '_change_epsilons'
    if args.filter_number is not None and 'mnist' in args.task:
        experiment_name += f'_filter{args.filter_number}'
    
    if args.debug:
        log_dir = Path('experiments') / 'debug' / 'test' / time.strftime("%Y-%m-%d/%H_%M_%S/")  
    else:
        log_dir = Path('experiments') / experiment_name / 'test' / time.strftime("%Y-%m-%d/%H_%M_%S/")  
    
    ds_cached_dir = Path('experiments') / experiment_name / 'data'
    log_dir.mkdir(parents=True, exist_ok=True)
    ds_cached_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir = log_dir
    args.ds_cached_dir = ds_cached_dir
    if args.task.startswith('gaussian2mnist'):
        args.dim = 1
    else:
        args.dim = 2

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main_worker(args)

def train(args, model, train_dl, optimizer, loss_fn, before_train=None, after_train=None):
    losses = 0
    for data in train_dl:
        if isinstance(data, list):
            training_data, time = data
        else:
            training_data, time = data, None
            
        training_data = training_data.squeeze().float().cpu()
        x, y = training_data[:, :-args.dim], training_data[:, -args.dim:]
        if before_train is not None:
            x = before_train(x)
        x = x.to(args.device)
        y = y.to(args.device)
        time = time.to(args.device) if time is not None else None
        
        if args.debug:
            print(x.shape, time.shape if time is not None else None)
            
        pred = model(x, time) if time is not None else model(x)
        if after_train is not None:
            pred = after_train(pred)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses += loss.item() / len(train_dl)
    return losses

def main_worker(args):
    console = Console(record=True, color_system='truecolor')
    pretty = Pretty(args.__dict__, expand_all=True)
    panel = Panel(pretty, title='Arguments', expand=False, highlight=True)
    console.log(panel)
    console.log(f"Saving to {Path.absolute(args.log_dir)}")

    model, before_train, after_train = get_model_before_after(args)
    ds_info = gen_ds(args) # dict of information about the dataset (nums_sub_ds)
    
    if args.checkpoint is not None:
        try:
            model.load_state_dict(torch.load(args.checkpoint))
        except:
            console.log(":warning-emoji: [bold red blink] load checkpoint failed [/]\ncheckpoint: {}\nUse random init model".format(args.checkpoint))
        console.log("load checkpoint from {}".format(args.checkpoint))
    console.log(f"Model {model.__class__.__name__} Parameters: {int(sum(p.numel() for p in model.parameters())/1e6)}M")

    loss_list = []
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    real_metadata = pickle.loads(open(args.ds_cached_dir / 'real_mean_std.pkl', 'rb').read())

    if args.scheduler == 'cos':
        console.log(f"Using CosineAnnealingLR")
        scheduler = OneCycleLR(optimizer, max_lr=args.lr, total_steps=ds_info['nums_sub_ds']*args.epoch_nums*args.iter_nums)
    else:
        console.log(f"Scheduling disabled")
        scheduler = None
        
    console.rule("[bold spring_green2 blink]Training")
    model.to(args.device)
    model.train() 
    with Progress(
            SpinnerColumn(spinner_name='moon'),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            transient=False,
        ) as progress:
        task1 = progress.add_task("[red]Training whole dataset (lr: X) (loss=X)", total=ds_info['nums_sub_ds']*args.epoch_nums)
        while not progress.finished:
            if ds_info['nums_sub_ds'] == 1:
                new_dl = read_ds_from_pkl(args, 
                                          real_metadata,
                                          args.ds_cached_dir / f"new_ds_0.pkl" 
                                          )
            for iter in range(ds_info['nums_sub_ds']*args.epoch_nums):
                if ds_info['nums_sub_ds'] > 1:
                    new_dl = read_ds_from_pkl(args, 
                                            real_metadata,
                                            args.ds_cached_dir / f"new_ds_{int(iter%ds_info['nums_sub_ds'])}.pkl" 
                                            )

                task2 = progress.add_task(f"[dark_orange]Training sub dataset {int(iter%ds_info['nums_sub_ds'])}", total=args.iter_nums)
                for _ in range(args.iter_nums):
                    now_loss = train(args, model ,new_dl, optimizer, loss_fn, before_train, after_train)
                    if scheduler is not None:
                        scheduler.step()
                    loss_list.append(now_loss)
                    cur_lr = optimizer.param_groups[-1]['lr']
                    progress.update(task2, advance=1)
                progress.update(task2, visible=False)
                progress.remove_task(task2)
                torch.save(model.state_dict(), args.log_dir / f'model_{model.__class__.__name__}_{int(iter)}.pth')
                progress.update(task1, advance=1, description="[red]Training whole dataset (lr: %2.5f) (loss=%2.5f)" % (cur_lr, now_loss))
                progress.log("[green]sub dataset %d finished; Loss: %2.5f" % (int(iter%ds_info['nums_sub_ds']), now_loss))
    
    console.rule("[bold bright_green blink]Finished Training")
    console.log("Final loss: %2.5f" % (loss_list[-1]))
    # Draw loss curve
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(loss_list)
    ax.set_title("Loss")
    fig.savefig(args.log_dir / 'loss.png')
    console.log("Loss curve saved to {}".format(args.log_dir / 'loss.png'))

    torch.save(model.state_dict(), args.log_dir / f'model_{model.__class__.__name__}_final.pth')
    console.log("Model saved to {}".format(args.log_dir / f'model_{model.__class__.__name__}_final.pth'))

if __name__ == '__main__':
    main()
    pass