from torch.utils.data import Dataset, DataLoader 

class BBdataset(Dataset):
    def __init__(self, data):
        self.data = data  
        
    def __len__(self):
        return len(self.data)  
    
    def __getitem__(self, index):
        return self.data[index, :]
    
    
def get_dataloader(raw_data, args):
    train_ds = BBdataset(raw_data)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    return train_dl