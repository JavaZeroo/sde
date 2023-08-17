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
            console.log(":warning-emoji: [bold red blink] load checkpoint failed [/]\ncheckpoint: {}\nUse random init model".format(args.checkpoint))
            
        console.log("load checkpoint from {}".format(args.checkpoint))
        
    console.log(f"Model {model.__class__.__name__} Parameters: {int(sum(p.numel() for p in model.parameters())/1e6)}M")
    
    loss_list = []
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    real_metadata = pickle.loads(open(args.ds_cached_dir / 'real_mean_std.pkl', 'rb').read())

    if args.scheduler == 'cosine':
        scheduler = OneCycleLR(optimizer, max_lr=args.lr, total_steps=60*args.epoch_nums*args.iter_nums)
    else:
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
        task1 = progress.add_task("[gold]Training whole dataset (lr: X) (loss=X)", total=60*args.epoch_nums)
        while not progress.finished:
            for iter in range(60*args.epoch_nums):
                new_dl = get_ds()
                new_ds = pickle.loads(open(args.ds_cached_dir / f'new_ds_{int(iter%60)}.pkl', 'rb').read())
                # 应该要删掉的
                new_ds = reverse_normalize_dataset(new_ds.metadata, ts=new_ds.ts, bridge=new_ds.bridge, drift=new_ds.drift, source=new_ds.source_sample, target=new_ds.target_sample)
                new_ds = normalize_dataset_with_metadata(real_metadata, **new_ds)
                new_ds = MNISTdataset(new_ds['ts'], new_ds['bridge'], new_ds['drift'], new_ds['source'], new_ds['target'])
                new_dl = DataLoader(new_ds, batch_size=args.batch_size, shuffle=True, num_workers=10)

                task2 = progress.add_task(f"[green]Training sub dataset {int(iter%60)}", total=args.iter_nums)
                for e in range(args.iter_nums):
                    now_loss = train(model ,new_dl, optimizer, scheduler, loss_fn, before_train, after_train)
                    loss_list.append(now_loss)
                    cur_lr = optimizer.param_groups[-1]['lr']
                    progress.update(task2, advance=1)
                progress.update(task2, visible=False)
                progress.remove_task(task2)
                torch.save(model.state_dict(), args.log_dir / f'model_{model.__class__.__name__}_{int(iter)}.pth')
                progress.update(task1, advance=1, description="[red]Training whole dataset (lr: %2.5f) (loss=%2.5f)" % (cur_lr, now_loss))

    # Draw loss curve
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(loss_list)
    ax.set_title("Loss")
    fig.savefig(args.log_dir / 'loss.png')









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
            
            x = F.pad(x, (2, 2, 2, 2, 0, 0, 0, 0), 'constant', 0)
            dydt = model(x.to(args.device)).cpu()
            dydt = dydt[:, :, 2:-2, 2:-2]
            
            pred_drift[i]=dydt
            dydt = reverse_normalize_dataset(real_metadata, bridge=dydt)['bridge']

            diffusion = sigma * torch.sqrt(dt) * torch.randn_like(dydt)

            pred_bridge[i+1] = pred_bridge[i] + dydt * dt + diffusion[:]

    plot_source_and_target_mnist(test_bridge[-1, :25], binary(pred_bridge[-1, :25]))

    save_gif_frame_mnist(pred_bridge[:, :25], args.log_dir, 'pred.gif')


if __name__ == '__main__':
    main()
    pass