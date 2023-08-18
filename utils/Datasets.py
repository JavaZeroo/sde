from torch.utils.data import Dataset, DataLoader 
from  torch import arange,  meshgrid, stack, cat, split, concat, unsqueeze

class BBdataset(Dataset):
    def __init__(self, data):
        self.data = data  
        
    def __len__(self):
        return len(self.data)  
    
    def __getitem__(self, index):
        return self.data[index, :]
    
    
class MNISTdataset(Dataset):
    def __init__(self, ts, bridge, drift, source, target, time_expand=True):
        self.ts = ts
        self.bridge = bridge
        self.drift = drift
        self.source = source
        self.target = target
        self.time_expand = time_expand
        
        lent,lensample,_,_,_ = bridge.shape
        # 创建一维索引
        i = arange(lent)
        j = arange(lensample)

        # 构建网格
        ii, jj = meshgrid(i, j, indexing='ij')

        # 重塑并组合
        index_list = stack((ii.flatten(), jj.flatten()), dim=1)

        # 转换为Python索引
        self.index_list = [tuple(index) for index in index_list.tolist()]

    def __len__(self):
        return len(self.index_list)
    
    def __getitem__(self, index):
        ti, samplei = self.index_list[index]
        start = self.source[samplei].unsqueeze(dim=0)
        if self.time_expand:
            times = self.ts[ti].repeat(28, 28).unsqueeze(dim=0).unsqueeze(dim=0)
        else:
            times = self.ts[ti]
        positions = self.bridge[ti, samplei, :, :].unsqueeze(dim=0)
        scores = self.drift[ti, samplei, :, :].unsqueeze(dim=0)
        # print(start.shape, times.shape, positions.shape, scores.shape)
        if self.time_expand:
            raw_data = concat([start, times, positions, scores], dim=1)
            return raw_data
        else:
            raw_data = concat([start, positions, scores], dim=1)
            return raw_data, times
            
    
    def to_dict(self):
        return {
            "ts": self.ts,
            "bridge": self.bridge,
            "drift": self.drift,
            "source": self.source,
            "target": self.target,
        }
        
def get_dataloader(raw_data, args):
    train_ds = BBdataset(raw_data)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    return train_dl