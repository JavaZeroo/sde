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
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
import time

from utils.Models import MLP
from utils.Datasets import BBdataset, MNISTdataset
from utils.utils import plot_source_and_target_mnist, binary, save_gif_frame_mnist
from utils.data_utils import gen_mnist_data, reverse_normalize_dataset, normalize_dataset_with_metadata
import argparse

import logging
from rich.logging import RichHandler

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

FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)
log = logging.getLogger("rich")

log.info(f"Saving to {log_dir}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log.info(f"Using device: {device}")

# model = MLP(input_dim=3, output_dim=1, hidden_layers=4, hidden_dim=256).to(device)
# from utils.unet import UNet
# model = UNet(in_channels=3, out_channels=1, init_features=24).to(device)
# os.system("source ~/add_proxy.sh")
if args.task == 'gaussian2minst':
    model = smp.Unet(
        encoder_name="efficientnet-b0",
        in_channels=3,
        encoder_depth=3, 
        decoder_channels=(64, 32, 16),
        classes=1,
        # decoder_attention_type="scse",
        )
else:
    model = MLP(
        input_dim=3, 
        output_dim=1, 
        hidden_layers=4, 
        hidden_dim=256
        )
    
if args.checkpoint is not None:
    model.load_state_dict(torch.load(args.checkpoint))
    log.info("load checkpoint from {}".format(args.checkpoint))
    
log.info(f"Model {model.__class__.__name__} Parameters: {int(sum(p.numel() for p in model.parameters())/1e6)}M")

model.to(device)

def train(model, train_dl, optimizer, scheduler, loss_fn, before_train=None, after_train=None):
    losses = 0
    for training_data in train_dl:
        training_data = training_data[:,0].float().cpu()
        x, y = training_data[:, :-1], training_data[:, -1:]
        if before_train is not None:
            # x = F.pad(x, (2, 2, 2, 2, 0, 0, 0, 0), 'constant', 0)
            x = before_train(x)
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        if after_train is not None:
            # pred = pred[:, :, 2:-2, 2:-2]
            pred = after_train(pred)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()
        losses += loss.item() / len(train_dl)
        
    return losses


  
loss_list = []
loss_fn = nn.MSELoss()
iter_nums = 1       # means each sub dataset will be trained for 1 epoch
epoch_nums = 1      # means the whole dataset will be trained for 1 epoch
batch_size = 8000
lr = 1e-4
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
real_metadata = pickle.loads(open(f'/home/ljb/sde/experiments/gaussian2minst/data/mnist_mean_std.pkl', 'rb').read())

if args.scheduler == 'cosine':
    scheduler = OneCycleLR(optimizer, max_lr=lr, total_steps=60*epoch_nums*iter_nums)
else:
    scheduler = None
    
with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        transient=False,
    ) as progress:
    task1 = progress.add_task("[red]Training whole dataset (lr: X) (loss=X)", total=60*epoch_nums)
    while not progress.finished:
        for iter in range(60*epoch_nums):
            new_ds = pickle.loads(open(f'experiments/gaussian2minst/data/new_ds_{int(iter%60)}.pkl', 'rb').read())
            # 应该要删掉的
            ret = reverse_normalize_dataset(new_ds.metadata, ts=new_ds.ts, bridge=new_ds.bridge, drift=new_ds.drift, source=new_ds.source_sample, target=new_ds.target_sample)
            ret = normalize_dataset_with_metadata(real_metadata, **ret)
            temp_ds = MNISTdataset(ret['ts'], ret['bridge'], ret['drift'], ret['source'], ret['target'])
            new_dl = DataLoader(new_ds, batch_size=batch_size, shuffle=True, num_workers=10)

            # iter_iterator = tqdm(range(iter_nums), desc="Training (lr: X)  (loss= X)", dynamic_ncols=True)
            model.train()
            task2 = progress.add_task(f"[green]Training sub dataset {int(iter%60)}", total=iter_nums)
            for e in range(iter_nums):
                now_loss = train(model ,new_dl, optimizer, scheduler, loss_fn)
                loss_list.append(now_loss)
                cur_lr = optimizer.param_groups[-1]['lr']
                # iter_iterator.set_description("Training (lr: %2.5f)  (loss=%2.5f)" % (cur_lr, now_loss))
                progress.update(task2, advance=e+1)
            progress.update(task2, visible=False)
            progress.remove_task(task2)
            torch.save(model.state_dict(), log_dir / f'model_unet_{int(iter)}.pth')
            progress.update(task1, advance=iter, description="[red]Training whole dataset (lr: %2.5f) (loss=%2.5f)" % (cur_lr, now_loss))


plt.plot(loss_list)
plt.title("Loss")
plt.savefig(log_dir / 'loss.png')

torch.save(model.state_dict(), log_dir / 'model_unet.pth')

test_ts, test_bridge, test_drift, test_source_sample, test_target_sample = gen_mnist_data(nums=1000)

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
        dydt = model(x.to(device)).cpu()
        dydt = dydt[:, :, 2:-2, 2:-2]
        
        pred_drift[i]=dydt
        dydt = reverse_normalize_dataset(real_metadata, bridge=dydt)['bridge']

        diffusion = sigma * torch.sqrt(dt) * torch.randn_like(dydt)

        pred_bridge[i+1] = pred_bridge[i] + dydt * dt + diffusion[:]

plot_source_and_target_mnist(test_bridge[-1, :25], binary(pred_bridge[-1, :25]))

save_gif_frame_mnist(pred_bridge[:, :25], log_dir, 'pred.gif')