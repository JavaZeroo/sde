import torch
import math

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
    train_set = torchvision.datasets.MNIST(root="./data/", train=True, download=True)
    result = train_set.data[:nb].to(device).view(-1, 1, 28, 28).float()
    return result