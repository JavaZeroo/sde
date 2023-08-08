# %%
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import Dataset, DataLoader 

torch.manual_seed(233)
np.random.seed(233)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# %%
mu = 0
sigma = 1
a = 10
T = 1
epsilon = 0.001
num_samples = 5

t = torch.arange(0, T+epsilon, epsilon)

y_mean = a * t
variance = sigma**2 * t * (1 - t/T)
std_dev = torch.sqrt(variance)

upper_bound = y_mean + 3 * std_dev
lower_bound = y_mean - 3 * std_dev

y_bridge = torch.zeros((t.shape[0], num_samples))
drift = torch.zeros((t.shape[0], num_samples))

# %%
for i in range(len(t) - 1):
    dt = t[i+1] - t[i]      # dt = epsilon
    dydt = (a * T - y_bridge[i]) / (T - t[i])
    drift[i, :] = dydt
    diffusion = sigma * torch.sqrt(dt) * torch.randn(num_samples)
    y_bridge[i+1] = y_bridge[i] + dydt * dt
    y_bridge[i+1, :] += diffusion
    
plt.figure(figsize=(10, 6))
plt.plot(t, y_bridge)
plt.plot(t, y_mean, label=r'<y>', linewidth=2, color='black', alpha=0.5)
plt.plot(t, lower_bound, color='orange', linewidth=1, label='Brownian Bridge lower_bound')
plt.plot(t, upper_bound, color='orange', linewidth=1, label='Brownian Bridge upper_bound')


plt.title('Brownian Bridge')
plt.xlabel('Time ($t$)')
plt.ylabel('$y(t)$')
plt.legend()
plt.grid(True)
plt.tight_layout()

# plt.savefig('outs/brownian_bridge.jpg', dpi=300)
plt.show()

# %%
class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        ret = self.fc1(x)
        ret = self.relu(ret)
        ret = self.fc2(ret)
        ret = self.relu(ret)
        return ret


# %%
model = MLP(1, 1).to(device)


# %%
drift.shape

# %%
dataset = drift[:len(drift)-1].reshape(-1,1)

# %%
dataset.shape

# %%
t.reshape(-1 ,1).repeat(num_samples)

# %%



