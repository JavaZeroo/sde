########################################################################
######  To learn the score of the transmat from the path of Brownian bridge
########################################################################
import math, argparse
import numpy as np
import six
import matplotlib.pyplot as plt
import torch, torchvision
from torch import nn

from numpy import exp, arange
from pylab import meshgrid, cm, imshow, contour, clabel, colorbar, axis, title, show

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


seed = 0
N = 200
M = 2*50

np.random.seed(seed)
path = '/Users/yangsikun/Desktop/DiffusionModels/NonlinearDiffusionProcesses/'


def sample_path_batch(M, N):
    B = np.empty((M, N), dtype=np.float32)
    R = np.empty((M, N), dtype=np.float32)
    # B[:, 0] = np.array([1, 2, 3, 4, 5]) * 0 #+ 5
    B[:, 0] = np.ones((1,M)) * 10
    # R[:,0] = B[:,0]
    dt = 1.0 / N
    dt_sqrt = np.sqrt(dt)
    # b = 5
    for n in six.moves.range(N - 1):
        t = n * dt
        # print('Remaining time:',1-t)
        # print('Value loss:',b-B[0,n])
        # xi = (np.random.randn(2,1)) * dt_sqrt# (np.random.randn(M)) * np.sqrt(t*b + (1-t)*t) * dt_sqrt#* dt_sqrt
        xi = (np.random.randn(M)) * dt_sqrt
        # B[:, n + 1] = B[:, n] + (np.sin(np.power(B[:, n],1))) * dt + xi#  B[:, n]+    + (1/(1 - t)) * (b-B[:, n])
        B[:, n + 1] = B[:, n] -(2+(np.sin(np.power(B[:, n],1)))) * dt + xi
        # R[:, n + 1] = (1 / (1 - t)) * (b - B[:, n]) #* dt
        #B[:, n + 1] = (1 / (1 - t)) * (b - B[:, n]) * dt + xi
        # B[:, n + 1] = B[:, n]  + xi #+ (1 / (1 - t)) * (b - B[:, n]) * dt + xi

    return B# ,R

# def sample_path_batch2(M, N):
#     dt = 1.0 / (N -1)                                                          #  changed from 1.0 / N
#     dt_sqrt = np.sqrt(dt)
#     B = np.empty((M, N), dtype=np.float32)
#     B[:, 0] = 10
#     for n in six.moves.range(N - 2):                                           # changed from "for n in six.moves.range(N - 1)"
#          t = n * dt
#          xi = np.random.randn(M) * dt_sqrt
#          B[:, n + 1] = B[:, n] * (1 - dt / (1 - t)) + xi
#     B[:, -1] = 0                                                               # added: set the last B to Zero
#     return B
plt.figure()
B = sample_path_batch(M, N)
#pyplot.semilogx(R[:,39:49].T)
plt.plot(B.T)
plt.grid()
plt.show() # plt.savefig(path + 'SampledPathOfDiffussionProcessWithNonlinearDrift.png')


###
initial_pt = B[0,0]
target_pts = B[:,-1]
# plt.show()


print(f'device {device}')

######################################################################de
def sample_path_batch(M, N, a, b):
    Ndt = 500*2
    Ndx = 100
    drift_values = torch.zeros((Ndt * Ndx, 3))

    B = np.empty((M, N), dtype=np.float32)
    Drift = np.empty((M, N), dtype=np.float32)
    B[:, 0] = a#np.array([1, 2, 3, 4, 5]) * 2

    dt = 1.0 / N
    dt_sqrt = np.sqrt(dt)
    # b = 5
    for n in six.moves.range(N - 1):
        t = n * dt
        xi = (np.random.randn(M)) * dt_sqrt
        Drift[:, n + 1] =  (np.sin(np.power(B[:, n],2)) + ((1 / (1 - t)) * (b - B[:, n])))
        B[:, n + 1] = B[:, n] + Drift[:, n + 1] * dt+ xi
    return B, Drift

def sample_condBBpath(nb):
    #target_value = 6.0
    #target_value_set = torch.tensor([0.0, 1.0, 2.0,3.0,4.0,5.0,6.0])
    target_value_set = torch.tensor(target_pts)
    N_samps = 100 * 10
    # X_0 = np.ones((1, N_samps)) * 0.0  # np.random.uniform(0,3,N_samps)
    # X_1 = np.ones((1, N_samps)) * target_value  # np.random.uniform(0,3,N_samps)
    Ndt = 200
    # Ndx = 100
    drift_values = torch.zeros((Ndt * N_samps, 4))
    # B = np.zeros((N_samps, Ndt))
    # Drift = np.zeros((N_samps, Ndt))
    # for isim in range(N_samps):
    #     # temp1, temp2\
    #     B[isim, :], Drift[isim, :] = sample_path_batch(1, Ndt, X_0[0, isim], X_1[0, isim])
    #     # B[isim, :], Drift[isim, :] = temp1, temp2
    # plt.plot(B.T)
    # plt.show()
    dt = 1.0 / Ndt
    dt_sqrt = np.sqrt(dt)
    for isim in range(N_samps):
        B = torch.empty((1, Ndt), dtype=torch.float32)
        Drift = torch.empty((1, Ndt), dtype=torch.float32)
        B[:, 0] = torch.tensor(initial_pt)
        for n in six.moves.range(Ndt - 1):
            t = n * dt
            xi = (np.random.randn(1)) * dt_sqrt
            Drift[:, n + 1] =  (((1 / (1 - t)) * (target_value_set[isim%6] - B[:, n])))
            B[:, n + 1] = B[:, n] + Drift[:, n + 1] * dt+ xi
            drift_values[n + isim * Ndt, 0] = B[:, n]# (xinput * target_value) / Ndx
            drift_values[n + isim * Ndt, 1] = t # n / Ndt
            drift_values[n + isim * Ndt, 2] = target_value_set[isim%6]
            drift_values[n + isim * Ndt, 3] = Drift[:, n + 1]# (target_value - (xinput * target_value) / Ndx) / (1 - n / Ndt)

    return drift_values

def sample_BBpath(nb):
    target_value = 6.0
    target_value_set = torch.tensor([0.0, 1.0, 2.0,3.0,4.0,5.0,6.0])
    N_samps = 100
    # X_0 = np.ones((1, N_samps)) * 0.0  # np.random.uniform(0,3,N_samps)
    # X_1 = np.ones((1, N_samps)) * target_value  # np.random.uniform(0,3,N_samps)
    Ndt = 200
    # Ndx = 100
    drift_values = torch.zeros((Ndt * N_samps, 3))
    # B = np.zeros((N_samps, Ndt))
    # Drift = np.zeros((N_samps, Ndt))
    # for isim in range(N_samps):
    #     # temp1, temp2\
    #     B[isim, :], Drift[isim, :] = sample_path_batch(1, Ndt, X_0[0, isim], X_1[0, isim])
    #     # B[isim, :], Drift[isim, :] = temp1, temp2
    # plt.plot(B.T)
    # plt.show()
    dt = 1.0 / Ndt
    dt_sqrt = np.sqrt(dt)
    for isim in range(N_samps):
        B = torch.empty((1, Ndt), dtype=torch.float32)
        Drift = torch.empty((1, Ndt), dtype=torch.float32)
        B[:, 0] = 0
        for n in six.moves.range(Ndt - 1):
            t = n * dt
            xi = (np.random.randn(1)) * dt_sqrt
            Drift[:, n + 1] =  (((1 / (1 - t)) * (target_value - B[:, n])))
            B[:, n + 1] = B[:, n] + Drift[:, n + 1] * dt+ xi
            drift_values[n + isim * Ndt, 0] = B[:, n]# (xinput * target_value) / Ndx
            drift_values[n + isim * Ndt, 1] = t# n / Ndt
            drift_values[n + isim * Ndt, 2] = Drift[:, n + 1]# (target_value - (xinput * target_value) / Ndx) / (1 - n / Ndt)

    return drift_values

def sample_region(nb):
    target_value = 6
    Ndt = 200
    Ndx = 100
    drift_values = torch.zeros((Ndt*Ndx,3))
    for n in range(Ndt):
        for xinput in range(Ndx):
            drift_values[n + xinput * Ndt, 0] = (xinput * target_value)/Ndx
            drift_values[n + xinput * Ndt, 1] = n/Ndt
            drift_values[n+xinput*Ndt, 2] = (target_value - (xinput * target_value)/Ndx) / (1 - n / Ndt)
    return drift_values

def sample_gaussian_mixture(nb):
    p, std = 0.3, 0.2
    result = torch.randn(nb, 1) * std
    result = result + torch.sign(torch.rand(result.size()) - p) / 2
    return result

def sample_ramp(nb):
    result = torch.min(torch.rand(nb, 1), torch.rand(nb, 1))
    return result

def sample_two_discs(nb):
    a = torch.rand(nb) * math.pi * 2
    b = torch.rand(nb).sqrt()
    q = (torch.rand(nb) <= 0.5).long()
    b = b * (0.3 + 0.2 * q)
    result = torch.empty(nb, 2)
    result[:, 0] = a.cos() * b - 0.5 + q
    result[:, 1] = a.sin() * b - 0.5 + q
    return result

def sample_disc_grid(nb):
    a = torch.rand(nb) * math.pi * 2
    b = torch.rand(nb).sqrt()
    N = 4
    q = (torch.randint(N, (nb,)) - (N - 1) / 2) / ((N - 1) / 2)
    r = (torch.randint(N, (nb,)) - (N - 1) / 2) / ((N - 1) / 2)
    b = b * 0.1
    result = torch.empty(nb, 2)
    result[:, 0] = a.cos() * b + q
    result[:, 1] = a.sin() * b + r
    return result

def sample_spiral(nb):
    u = torch.rand(nb)
    rho = u * 0.65 + 0.25 + torch.rand(nb) * 0.15
    theta = u * math.pi * 3
    result = torch.empty(nb, 2)
    result[:, 0] = theta.cos() * rho
    result[:, 1] = theta.sin() * rho
    return result

def sample_mnist(nb):
    train_set = torchvision.datasets.MNIST(root = './data/', train = True, download = True)
    result = train_set.data[:nb].to(device).view(-1, 1, 28, 28).float()
    return result

def sample_ring(nb):
    r = 1
    X = 0
    Y = 0
    # b = 0.001
    a = np.linspace((X - r), (X + r), 2000) #+ np.random.normal(0, 0.1, 2000)
    b = np.sqrt(pow(r, 2) - (a - X) * (a - X)) + Y + np.random.normal(0, 0.001, 2000)
    c = -np.sqrt(pow(r, 2) - (a - X) * (a - X)) + Y + np.random.normal(0, 0.001, 2000)
    a = a + + np.random.normal(0, 0.1, 2000)
    a = np.concatenate([a, a])
    b = np.concatenate([b, c])
    result = torch.empty(len(a), 2)
    result[:, 0] = torch.from_numpy(a)
    result[:, 1] = torch.from_numpy(b)#theta.sin() * rho
    return result

samplers = {
    'BBpath': sample_BBpath,
    'condBBpath': sample_condBBpath,
    'BBDrift': sample_region,
    'gaussian_mixture': sample_gaussian_mixture,
    'ramp': sample_ramp,
    'two_discs': sample_two_discs,
    'disc_grid': sample_disc_grid,
    'spiral': sample_spiral,
    'mnist': sample_mnist,
    'ring':sample_ring,
}

######################################################################

parser = argparse.ArgumentParser(
    description = '''A minimal implementation of Jonathan Ho, Ajay Jain, Pieter Abbeel
"Denoising Diffusion Probabilistic Models" (2020)
https://arxiv.org/abs/2006.11239''',

    formatter_class = argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--seed',
                    type = int, default = 0,
                    help = 'Random seed, < 0 is no seeding')

parser.add_argument('--nb_epochs',
                    type = int, default = 300,#1000/500,
                    help = 'How many epochs')

parser.add_argument('--batch_size',
                    type = int, default = 25,
                    help = 'Batch size')

parser.add_argument('--nb_samples',
                    type = int, default = 25000,
                    help = 'Number of training examples')

parser.add_argument('--learning_rate',
                    type = float, default = 1e-3,
                    help = 'Learning rate')

parser.add_argument('--ema_decay',
                    type = float, default = 0.9999,
                    help = 'EMA decay, <= 0 is no EMA')

data_list = ', '.join( [ str(k) for k in samplers ])

# parser.add_argument('--data',
#                     type = str, default = 'gaussian_mixture',
#                     help = f'Toy data-set to use: {data_list}')
parser.add_argument('--data',
                    type = str, default = 'condBBpath',#'BBDrift',
                    help = f'Toy data-set to use: {data_list}')

args = parser.parse_args()

if args.seed >= 0:
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

######################################################################

class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.mem = { }
        with torch.no_grad():
            for p in model.parameters():
                self.mem[p] = p.clone()

    def step(self):
        with torch.no_grad():
            for p in self.model.parameters():
                self.mem[p].copy_(self.decay * self.mem[p] + (1 - self.decay) * p)

    def copy_to_model(self):
        with torch.no_grad():
            for p in self.model.parameters():
                p.copy_(self.mem[p])

######################################################################

class ConvNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        ks, nc = 5, 64

        self.core = nn.Sequential(
            nn.Conv2d(in_channels, nc, ks, padding = ks//2),
            nn.ReLU(),
            nn.Conv2d(nc, nc, ks, padding = ks//2),
            nn.ReLU(),
            nn.Conv2d(nc, nc, ks, padding = ks//2),
            nn.ReLU(),
            nn.Conv2d(nc, nc, ks, padding = ks//2),
            nn.ReLU(),
            nn.Conv2d(nc, nc, ks, padding = ks//2),
            nn.ReLU(),
            nn.Conv2d(nc, out_channels, ks, padding = ks//2),
        )

    def forward(self, x):
        return self.core(x)

######################################################################
# Data

try:
    train_input = samplers[args.data](args.nb_samples).to(device)
except KeyError:
    print(f'unknown data {args.data}')
    exit(1)

# train_mean, train_std = train_input.mean(), train_input.std()

######################################################################
# Model
if train_input.dim() == 2:
    nh = 256

    model = nn.Sequential(
        # nn.Linear(train_input.size(1) + 1, nh),
        nn.Linear(3, nh),
        nn.ReLU(),
        nn.Linear(nh, nh),
        nn.ReLU(),
        nn.Linear(nh, nh),
        nn.ReLU(),
        nn.Linear(nh, nh),
        nn.ReLU(),
        nn.Linear(nh, nh),
        nn.ReLU(),
        nn.Linear(nh, 1),
    )

elif train_input.dim() == 4:

    model = ConvNet(train_input.size(1) + 1, train_input.size(1))

model.to(device)

print(f'nb_parameters {sum([ p.numel() for p in model.parameters() ])}')

######################################################################
# Generate

# def generate(size, alpha, alpha_bar, sigma, model, train_mean, train_std):
#
#     with torch.no_grad():
#
#         x = torch.randn(size, device = device)
#
#         for t in range(T-1, -1, -1):
#             z = torch.zeros_like(x) if t == 0 else torch.randn_like(x)
#             input = torch.cat((x, torch.full_like(x[:,:1], t / (T - 1) - 0.5)), 1)
#             x = 1/torch.sqrt(alpha[t]) \
#                 * (x - (1-alpha[t]) / torch.sqrt(1-alpha_bar[t]) * model(input)) \
#                 + sigma[t] * z
#
#         x = x * train_std + train_mean
#
#         return x

######################################################################
# Train

# T = 1000
# beta = torch.linspace(1e-4, 0.02, T, device = device)
# alpha = 1 - beta
# alpha_bar = alpha.log().cumsum(0).exp()
# sigma = beta.sqrt()

ema = EMA(model, decay = args.ema_decay) if args.ema_decay > 0 else None

for k in range(args.nb_epochs):

    acc_loss = 0
    optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)

    for x0 in train_input.split(args.batch_size):
        input = x0[:,0:3]
        loss = (model(input) - x0[:,3]).pow(2).mean()
        acc_loss += loss.item() * x0.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ema is not None: ema.step()

    print(f'{k} {acc_loss}')# / train_input.size(0)}')

if ema is not None: ema.copy_to_model()

######################################################################
model.eval()
# N = 100
# target_value = 6
# outvalue = np.zeros((100,200))
# for n in range(200):
#     for xinput in range(100):
#         input = torch.tensor([(xinput * target_value) / 100, n/200])  # input = torch.tensor([n/N, 0])
#         input = input.float()
#         outvalue[xinput,n] = model(input)
# im = plt.imshow(outvalue, cmap=cm.RdBu)  # drawing the function
# plt.colorbar(im)
# plt.title('NN drift value')
# show()

# a = 6.0
# for t_ind in range(10):
#     x = np.arange(0, 100, 1)
#     plt.figure(t_ind)
#     t = t_ind/10
#     T = 1.0
#     y = (a - (x * a) / 100) / (T - t)
#     plt.plot(x,y,'b.-')
#     out = torch.zeros((1,100))
#     for isim in range(100):
#         input = torch.tensor([(x[isim] * a)/100, t])
#         input = input.float()
#         out[0,isim] = model(input)
#     # y = Foellmer()#(a-x)/(T-t)
#     y = out.detach().numpy()
#     y = y.reshape(100,)
#     plt.title(t_ind)
#     plt.plot(x,y,'r.-')
#     plt.show()

##########################
############################################################################
def sample_path_nn(M, N, a,b):
    B = torch.empty((M, N), dtype=torch.float32)
    D = torch.empty((M, N), dtype=torch.float32)
    B[:,0] = torch.tensor(a)#np.array([1, 2, 3, 4, 5]) * 2
    D[:,0] = torch.tensor(0)

    dt = 1.0 / N
    dt_sqrt = np.sqrt(dt)

    for samp in range(1):
        for n in six.moves.range(N - 1):
            t = n * dt
            xi = (torch.randn(M)) * dt_sqrt
            input = torch.tensor([B[samp, n],n/N,b])
            input = input.float()
            D[samp, n + 1] = model(input) * dt
            B[samp, n + 1] = D[samp, n + 1] + xi + B[samp, n]# + temp + xi

    return B,D

N = 100 * 10
M = 1
N_eval_samps = 1#30
# for target_ind in range(len(target_pts)):
#     print(target_ind)
#     target_value = target_pts[target_ind]
#     # target_value = 5.0
#     ###
#     # a = 5.0
#     # for t_ind in range(10):
#     #     x = np.arange(0, 100, 1)
#     #     plt.figure(t_ind)
#     #     t = t_ind / 10
#     #     T = 1.0
#     #     y = (a - (x * a) / 100) / (T - t)
#     #     plt.plot(x, y, 'b.-')
#     #     out = torch.zeros((1, 100))
#     #     for isim in range(100):
#     #         input = torch.tensor([(x[isim] * a) / 100, t, target_value])
#     #         input = input.float()
#     #         out[0, isim] = model(input)
#     #     # y = Foellmer()#(a-x)/(T-t)
#     #     y = out.detach().numpy()
#     #     y = y.reshape(100, )
#     #     plt.title(t_ind)
#     #     plt.plot(x, y, 'r.-')
#     #     plt.show()
#     # ###
#     B_eval = torch.zeros((N_eval_samps,N))
#     D_eval = torch.zeros((N_eval_samps,N))
#     for isim in range(N_eval_samps):
#         B_eval[isim,:],D_eval[isim,:] = sample_path_nn(M, N, initial_pt, target_value)#isim%10) #X_0[isim])
#
#     B_eval = B_eval.detach().numpy()
#     plt.figure()
#     plt.plot(B_eval.T)
#     # plt.ylim(0, 10)
#     plt.title('conditioning on y='+str(target_value))
#     plt.grid()
#     plt.savefig(path + 'Simulate_a_predefined_diffusion_bridge_with_NN_drift'+str(target_ind)+'.png')
#     plt.show()

#####
B_eval = torch.zeros((len(target_pts),N))
D_eval = torch.zeros((len(target_pts),N))
for target_ind in range(len(target_pts)):
    target_value = target_pts[target_ind]
    for isim in range(N_eval_samps):
        B_eval[target_ind,:],D_eval[target_ind,:] = sample_path_nn(M, N, initial_pt, target_value)#isim%10) #X_0[isim])
B_eval = B_eval.detach().numpy()

plt.figure()
plt.plot(B_eval.T)
# plt.ylim(0, 10)
plt.title('Simulate a predefined diffusion process with NN drift')
plt.grid()
plt.show()
# plt.savefig(path + 'Simulate_a_predefined_diffusion_process_with_NN_drift.png')
print('d')
# plt.show()