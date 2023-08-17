import torch
import torchvision
from utils.Datasets import MNISTdataset
import pickle
from rich.progress import track

def gen_bridge_2d(x, y, ts, T, num_samples):
    """
    For Gaussian distribution. 
    """
    sigma = 1
    bridge = torch.zeros((ts.shape[0], num_samples, 2))
    drift = torch.zeros((ts.shape[0], num_samples, 2))
    bridge[0] = x
    for i in range(len(ts) - 1):
        dt = ts[i+1] - ts[i]
        dydt = (y - bridge[i]) / (T - ts[i])
        drift[i, :] = dydt
        diffusion = sigma * torch.sqrt(dt) * torch.randn(num_samples, 2)
        bridge[i+1] = bridge[i] + dydt * dt
        bridge[i+1, :] += diffusion
    return bridge, drift

# 主函数
def gen_2d_data(source_dist, target_dist, num_samples=1000, epsilon=0.001, T=1):
    """
    For Gaussian distribution, source_dist and target_dist are both (mean, std)
    """
    if not isinstance(num_samples, int):
        num_samples = int(num_samples)
    ts = torch.arange(0, T+epsilon, epsilon)
    source_dist = torch.Tensor(source_dist)
    target_dist = torch.Tensor(target_dist)
    bridge, drift = gen_bridge_2d(source_dist, target_dist, ts, T=T, num_samples=num_samples)
    return ts, bridge, drift, source_dist, target_dist


def normalize_dataset(ts, bridge, drift, source, target):
    """
    Normalize the dataset
    """
    mean_ts = torch.mean(ts, dim=0)
    std_ts = torch.std(ts, dim=0)
    
    ret_ts = (ts - mean_ts) / std_ts
    
    mean_bridge = torch.mean(bridge, dim=0)
    std_bridge = torch.std(bridge, dim=0)
    
    ret_bridge = (bridge - mean_bridge) / std_bridge
    
    mean_drift = torch.mean(drift, dim=0)
    std_drift = torch.std(drift, dim=0)
    
    ret_drift = (drift - mean_drift) / std_drift
    
    mean_source = torch.mean(source, dim=0)
    std_source = torch.std(source, dim=0)
    
    ret_source = (source - mean_source) / std_source
    
    mean_target = torch.mean(target, dim=0)
    std_target = torch.std(target, dim=0)
    
    ret_target = (target - mean_target) / std_target
    
    metadata = {
        'ts': (mean_ts, std_ts),
        'bridge': (mean_bridge, std_bridge),
        'drift': (mean_drift, std_drift),
        'source': (mean_source, std_source),
        'target': (mean_target, std_target)
    }
    
    retdata = {
        'ts': ret_ts,
        'bridge': ret_bridge,
        'drift': ret_drift,
        'source': ret_source,
        'target': ret_target
    }
    
    return retdata, metadata

def reverse_normalize_dataset(metadata, ts=None, bridge=None, drift=None, source=None, target=None):
    """
    Reverse normalize the dataset using the metadata
    """
    mean_ts, std_ts = metadata['ts']
    mean_bridge, std_bridge = metadata['bridge']
    mean_drift, std_drift = metadata['drift']
    mean_source, std_source = metadata['source']
    mean_target, std_target = metadata['target']
    
    ret_ts = ts * std_ts + mean_ts if ts is not None else None
    ret_bridge = bridge * std_bridge + mean_bridge if bridge is not None else None
    ret_drift = drift * std_drift + mean_drift if drift is not None else None
    ret_source = source * std_source + mean_source  if source is not None else None
    ret_target = target * std_target + mean_target if target is not None else None

    
    retdata = {
        'ts': ret_ts,
        'bridge': ret_bridge,
        'drift': ret_drift,
        'source': ret_source,
        'target': ret_target
    }
    
    return retdata


def normalize_dataset_with_metadata(metadata, ts=None, bridge=None, drift=None, source=None, target=None):
    """
    Normalize the dataset using the metadata
    """
    mean_ts, std_ts = metadata['ts']
    mean_bridge, std_bridge = metadata['bridge']
    mean_drift, std_drift = metadata['drift']
    mean_source, std_source = metadata['source']
    mean_target, std_target = metadata['target']
    
    ret_ts = (ts - mean_ts) / std_ts if ts is not None else None
    ret_bridge = (bridge - mean_bridge) / std_bridge if bridge is not None else None
    ret_drift = (drift - mean_drift) / std_drift if drift is not None else None
    ret_source = (source - mean_source) / std_source  if source is not None else None
    ret_target = (target - mean_target) / std_target if target is not None else None
    
    retdata = {
        'ts': ret_ts,
        'bridge': ret_bridge,
        'drift': ret_drift,
        'source': ret_source,
        'target': ret_target
    }
    
    return retdata


def gen_mnist_array_in_order(range=(0, 1000)):
    """
    Generate MNIST array in order
    """
    train_ds = torchvision.datasets.MNIST(
        root="./data/", 
        train=True, 
        download=True
        )
    target = train_ds.data.view(-1, 1, 28, 28).float()
    
    # random choice nums samples
    target = target[range[0]:range[1]]
    source = torch.randn_like(target)
    return source, target

def gen_mnist_array(nums=25):
    """
    Generate MNIST array and random choice nums samples
    """
    train_ds = torchvision.datasets.MNIST(
        root="./data/", 
        train=True, 
        download=True
        )
    target = train_ds.data.view(-1, 1, 28, 28).float()
    
    # random choice nums samples
    target = target[torch.randperm(target.shape[0])[:nums]]

    source = torch.randn_like(target)
    return source, target

def gen_bridge(x, y, ts, T):
    """
    Generate bridge process from x to y
    """
    sigma=1
    bridge = torch.zeros((len(ts), *x.shape))
    drift = torch.zeros((len(ts), *x.shape))
    bridge[0] = x # Initial value
    for i in range(len(ts) - 1):
        dt = ts[i+1] - ts[i]      # dt = epsilon
        dydt = (y - bridge[i]) / (T - ts[i])
        drift[i, :] = dydt
        diffusion = sigma * torch.sqrt(dt) * torch.randn_like(x)
        bridge[i+1] = bridge[i] + dydt * dt
        bridge[i+1, :] += diffusion
    return bridge, drift

def gen_mnist_data_in_order(range=(0, 1000), change_epsilons=False):
    """
    Generate MNIST dataset in order
    """
    source, target = gen_mnist_array_in_order(range)
    epsilon = 0.001
    T = 1
    ts = torch.arange(0, T+epsilon, epsilon)
    
    T = 1
    if change_epsilons:
        epsilon1 = 0.001
        epsilon2 = 0.0001

        t1 = torch.arange(0, 0.91, epsilon1)
        t2 = torch.arange(0.91, T, epsilon2)
        ts = torch.concatenate((t1, t2))
    else:
        epsilon = 0.001
        ts = torch.arange(0, T+epsilon, epsilon)
    
    bridge, drift = gen_bridge(source, target, ts, T)
    return ts, bridge, drift, source, target

def gen_mnist_data(nums=100, change_epsilons=False):
    """
    Generate MNIST dataset
    """
    source, target = gen_mnist_array(nums)
    
    T = 1
    if change_epsilons:
        epsilon1 = 0.001
        epsilon2 = 0.0001

        t1 = torch.arange(0, 0.91, epsilon1)
        t2 = torch.arange(0.91, T, epsilon2)
        ts = torch.concatenate((t1, t2))
    else:
        epsilon = 0.001
        ts = torch.arange(0, T+epsilon, epsilon)
        
    
    bridge, drift = gen_bridge(source, target, ts, T)
    return ts, bridge, drift, source, target


def preprocess_mnist_data(args):
    
    # check data pickle file
    for i in track(range(60), description="Preprocessing dataset"):
        if (args.log_dir / f'data/new_ds_{i}.pkl').exists():
            break
        ts, bridge, drift, source, target = gen_mnist_data_in_order((i*1000, (i+1)*1000))
        _, metadata = normalize_dataset(ts, bridge, drift, source, target)
        new_ds = MNISTdataset(ts, bridge, drift, source, target)
        new_ds.metadata = metadata
        pickle.dump(new_ds, open(args.log_dir / f'data/new_ds_{i}.pkl', 'wb'))



def get_data(args):
    if args.task == 'gaussian2minst':
        ts, bridge, drift, source, target = gen_mnist_data(nums=25)
    # ts, bridge, drift, source, target, ret = normalize_dataset(ts, bridge, drift, source, target)

