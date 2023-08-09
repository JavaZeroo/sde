from torch import nn
import torch

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=2, hidden_dim=64):
        super(MLP, self).__init__()
        
        self.fcin = nn.Linear(input_dim, hidden_dim)
        self.fcs = nn.ModuleList()
        for i in range(hidden_layers):
            self.fcs.append(nn.Linear(hidden_dim, hidden_dim))
        self.fcout = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        ret = self.fcin(x)
        ret = self.relu(ret)
        for fc in self.fcs:
            ret = fc(ret)
            ret = self.relu(ret)
        ret = self.fcout(ret)
        return ret
    
def get_model(args):
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")

    model = MLP(input_dim=args.input_dim, output_dim=args.output_dim, hidden_layers=args.hidden_layers, hidden_dim=args.hidden_dim).to(args.device)

    # 如果有多个 GPU，则使用 DataParallel
    if not args.single_gpu and  torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(args.device)

    return model