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
import time as tt

from utils.Datasets import BBdataset, MNISTdataset
from utils.utils import plot_source_and_target_mnist, binary, save_gif_frame_mnist
from utils.data_utils import gen_mnist_data, reverse_normalize_dataset, normalize_dataset_with_metadata, gen_ds
from utils.model_utils import get_model_before_after
import argparse

def check_model_task(args):
    if args.task == 'gaussian2mnist':
        assert args.model in ['tunet++', 'unet++', 'unet']
        args.time_expand = False
    else:
        assert args.model in ['mlp', 'unet++', 'unet']
        args.time_expand = True


def main():
    parser = argparse.ArgumentParser(description='Train SDE')
    
    parser.add_argument('--task', type=str, default='gaussian2mnist', required=True)
    parser.add_argument('--model', type=str, default='unet++', required=True)
    parser.add_argument('--checkpoint', type=str, default=None, required=True)
    
    parser.add_argument('--seed', type=int, default=233)
    parser.add_argument('--change_epsilons', action='store_true')
    parser.add_argument('--scheduler', type=str, default=None)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--iter_nums', type=int, default=1)
    parser.add_argument('--epoch_nums', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=8000)
    parser.add_argument('-n','--normalize', action='store_true')
    parser.add_argument('--tarined_data', action='store_true')
    parser.add_argument('--filter_number', type=int)

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
    
    
    log_dir = Path('experiments') / experiment_name / 'test' / tt.strftime("%Y-%m-%d/%H_%M_%S/")  
    ds_cached_dir = Path('experiments') / experiment_name / 'data'
    log_dir.mkdir(parents=True, exist_ok=True)
    ds_cached_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir = log_dir
    args.ds_cached_dir = ds_cached_dir

    if args.task == 'gaussian2mnist':
        args.dim = 1
    else:
        args.dim = 2
        
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main_worker(args)

def main_worker(args):
    console = Console(record=True, color_system='truecolor')
    pretty = Pretty(args.__dict__, expand_all=True)
    panel = Panel(pretty, title='Arguments', expand=False, highlight=True)
    console.log(panel)
    console.log(f"Saving to {Path.absolute(args.log_dir)}")

    model, before_train, after_train = get_model_before_after(args)
    # console.log(model)
    if args.checkpoint is not None:
        try:
            model.load_state_dict(torch.load(args.checkpoint))
        except:
            console.log(":warning-emoji: [bold red blink] load checkpoint failed [/]\ncheckpoint: {}\nExit".format(args.checkpoint))
            return None
        console.log("load checkpoint from {}".format(args.checkpoint))  
    console.log(f"Model {model.__class__.__name__} Parameters: {int(sum(p.numel() for p in model.parameters())/1e6)}M")
    
    real_metadata = pickle.loads(open(args.ds_cached_dir / 'real_mean_std.pkl', 'rb').read())

    if args.tarined_data:
        ds_cached_files = [f for f in args.ds_cached_dir.iterdir() if f.name.startswith('new_ds_')]
        temp_ds = pickle.loads(open(ds_cached_files[0], 'rb').read())
        data = {
            "ts": temp_ds.ts,
            "bridge": temp_ds.bridge,
            "drift": temp_ds.drift,
            "source": temp_ds.source,
        }
        data = reverse_normalize_dataset(temp_ds.metadata, **data)
        # data = normalize_dataset_with_metadata(real_metadata, **data)
        test_ts = data['ts']
        test_bridge = data['bridge']
        test_drift = data['drift']
        test_source = data['source']
    else:
        test_ts, test_bridge, test_drift, test_source, _ = gen_mnist_data(nums=1000)

    pred_bridge = torch.zeros_like(test_bridge)
    pred_drift = torch.zeros_like(test_drift)

    pred_bridge[0, :] = test_source
    # model.eval()
    
    sigma=1
    console.rule("[bold deep_sky_blue1 blink]Testing")
    with Progress(
            SpinnerColumn(spinner_name='moon'),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            transient=False,
        ) as progress:
        task1 = progress.add_task("[gold]Predicting", total=len(test_ts) - 1)
        with torch.no_grad():
            for i in range(len(test_ts) - 1):
                dt = (test_ts[i+1] - test_ts[i])
                test_source_reshaped = test_source
                if args.time_expand:
                    test_ts_reshaped = test_ts[i].repeat(test_source.shape[0]).reshape(-1, 1, 1, 1).repeat(1, 1, 28, 28)
                else:
                    test_ts_reshaped = torch.unsqueeze(test_ts[i], dim=0).T
                pred_bridge_reshaped = pred_bridge[i]

                ret = normalize_dataset_with_metadata(real_metadata, source=test_source_reshaped, ts=test_ts_reshaped, bridge=pred_bridge_reshaped)
                test_ts_reshaped = ret['ts']
                pred_bridge_reshaped = ret['bridge']
                test_source_reshaped = ret['source']
                if args.time_expand:
                    x = torch.concat([test_source_reshaped, test_ts_reshaped, pred_bridge_reshaped], axis=1)
                    time = None
                else:
                    x = torch.concat([test_source_reshaped, pred_bridge_reshaped], axis=1)
                    time = test_ts_reshaped.to(args.device)
                if before_train is not None:
                    x = before_train(x)

                x = x.to(args.device)
                model = model.to(args.device)
                dydt = model(x, time) if time is not None else model(x)
                dydt = dydt.cpu()
                if after_train is not None:
                    dydt = after_train(dydt)    
                pred_drift[i]=dydt
                dydt = reverse_normalize_dataset(real_metadata, bridge=dydt)['bridge']

                diffusion = sigma * torch.sqrt(dt) * torch.randn_like(dydt)

                pred_bridge[i+1] = pred_bridge[i] + dydt * dt + diffusion[:]
                progress.update(task1, advance=1)

    plot_source_and_target_mnist(test_bridge[-1, :25], binary(pred_bridge[-1, :25]), save_path=args.log_dir / 'source_and_target.jpg')

    save_gif_frame_mnist(pred_bridge[:, :25], args.log_dir, 'pred.gif')

    save_gif_frame_mnist(pred_bridge[:, :25], args.log_dir, 'pred_norm.gif', norm=True)
    

if __name__ == '__main__':
    main()
    pass