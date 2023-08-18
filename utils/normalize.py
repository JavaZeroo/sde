import torch
import pickle
from tqdm import tqdm

def calculate_total_mean_std(mean, std):
    # 计算drift的总均值和总标准差
    total_mean = mean.mean(dim=0)
    total_std = torch.sqrt(((std ** 2).mean(dim=0) + (mean - total_mean) ** 2).mean(dim=0))
    return total_mean, total_std

def get_total_mean_std(args):
    if (args.ds_cached_dir / 'real_mean_std.pkl').exists():
        return pickle.loads(open(args.ds_cached_dir / 'real_mean_std.pkl', 'rb').read())
    ds_cached_files = [f for f in args.ds_cached_dir.iterdir() if f.name.startswith('new_ds_')]
    ds_cached_files.sort(key=lambda x: int(x.name.split('_')[-1].split('.')[0]))
    nums_ds = len(ds_cached_files)
    
    temp_ds = pickle.loads(open(ds_cached_files[0], 'rb').read())
    ts_mean, ts_std  = temp_ds.metadata['ts']
    bridge_mean, bridge_std = temp_ds.metadata['bridge']
    drift_mean, drift_std = temp_ds.metadata['drift']
    source_mean, source_std = temp_ds.metadata['source']
    target_mean, target_std = temp_ds.metadata['target']
    if nums_ds == 1:
        ret = {
            'ts': (ts_mean, ts_std),
            'bridge': (bridge_mean, bridge_std),
            'drift': (drift_mean, drift_std),
            'source': (source_mean, source_std),
            'target': (target_mean, target_std),
        }
        return ret

    all_ts_mean = torch.zeros(nums_ds, *ts_mean.shape)
    all_ts_std = torch.zeros(nums_ds, *ts_std.shape)

    all_bridge_mean = torch.zeros(nums_ds, *bridge_mean.shape)
    all_bridge_std = torch.zeros(nums_ds, *bridge_std.shape)

    all_drift_mean = torch.zeros(nums_ds, *drift_mean.shape)
    all_drift_std = torch.zeros(nums_ds, *drift_std.shape)

    all_source_mean = torch.zeros(nums_ds, *source_mean.shape)
    all_source_std = torch.zeros(nums_ds, *source_std.shape)

    all_target_mean = torch.zeros(nums_ds, *target_mean.shape)
    all_target_std = torch.zeros(nums_ds, *target_std.shape)

    all_ts_mean[0] = ts_mean
    all_ts_std[0] = ts_std

    all_bridge_mean[0] = bridge_mean
    all_bridge_std[0] = bridge_std

    all_drift_mean[0] = drift_mean
    all_drift_std[0] = drift_std

    all_source_mean[0] = source_mean
    all_source_std[0] = source_std

    all_target_mean[0] = target_mean
    all_target_std[0] = target_std
    
    for i in tqdm(range(1, nums_ds)):
        i_ds = pickle.loads(open(ds_cached_files[i], 'rb').read())
        i_ts_mean, i_ts_std  = i_ds.metadata['ts']
        i_bridge_mean, i_bridge_std = i_ds.metadata['bridge']
        i_drift_mean, i_drift_std = i_ds.metadata['drift']
        i_source_mean, i_source_std = i_ds.metadata['source']
        i_target_mean, i_target_std = i_ds.metadata['target']

        all_ts_mean[i] = i_ts_mean
        all_ts_std[i] = i_ts_std

        all_bridge_mean[i] = i_bridge_mean
        all_bridge_std[i] = i_bridge_std

        all_drift_mean[i] = i_drift_mean
        all_drift_std[i] = i_drift_std

        all_source_mean[i] = i_source_mean
        all_source_std[i] = i_source_std

        all_target_mean[i] = i_target_mean
        all_target_std[i] = i_target_std
        
    ret_ts_mean, ret_ts_std = calculate_total_mean_std(all_ts_mean, all_ts_std)
    ret_bridge_mean, ret_bridge_std = calculate_total_mean_std(all_bridge_mean, all_bridge_std)
    ret_drift_mean, ret_drift_std = calculate_total_mean_std(all_drift_mean, all_drift_std)
    ret_source_mean, ret_source_std = calculate_total_mean_std(all_source_mean, all_source_std)
    ret_target_mean, ret_target_std = calculate_total_mean_std(all_target_mean, all_target_std)

    ret = {
        'ts': (ret_ts_mean, ret_ts_std),
        'bridge': (ret_bridge_mean, ret_bridge_std),
        'drift': (ret_drift_mean, ret_drift_std),
        'source': (ret_source_mean, ret_source_std),
        'target': (ret_target_mean, ret_target_std),
    }
    
    pickle.dump(ret, open(args.ds_cached_dir / 'real_mean_std.pkl', 'wb'))
    return ret