import torch

from config import torch_device
from two_moons_wiqvist_numpy import mean_radius, sd_radius, baseoffset
from sbi.utils import BoxUniform

prior_torch = BoxUniform(low=-2 * torch.ones(2), high=2 * torch.ones(2), device=torch_device)


def simulator_torch(theta):
    theta = theta.reshape(-1)
    a = torch.tensor(torch.pi * (torch.rand(1) - 0.5), device=torch_device)
    r = mean_radius + torch.randn(1, device=torch_device) * sd_radius
    p = torch.tensor([r * torch.cos(a) + baseoffset, r * torch.sin(a)], device=torch_device)
    ang = torch.tensor([-torch.pi / 4.0], device=torch_device)
    c = torch.cos(ang)
    s = torch.sin(ang)
    z0 = c * theta[0] - s * theta[1]
    z1 = s * theta[0] + c * theta[1]
    return p + torch.tensor([-torch.abs(z0), z1], device=torch_device)


def simulator_torch_snpla(theta):
    theta = theta.reshape(-1)
    a = torch.tensor(torch.pi * (torch.rand(1) - 0.5), device=torch_device)
    r = mean_radius + torch.randn(1, device=torch_device) * sd_radius
    p = torch.tensor([r * torch.cos(a) + baseoffset, r * torch.sin(a)], device=torch_device)
    ang = torch.tensor([-torch.pi / 4.0], device=torch_device)
    c = torch.cos(ang)
    s = torch.sin(ang)
    z0 = c * theta[0] - s * theta[1]
    z1 = s * theta[0] + c * theta[1]
    out = p + torch.tensor([-torch.abs(z0), z1], device=torch_device)
    return out.reshape(1, 2)


def simulator_torch_batched(theta):
    n_sim, data_dim = theta.shape
    x_samples = torch.zeros((n_sim, data_dim))

    for i in range(n_sim):
        x_samples[i, ] = simulator_torch(theta[i,])

    return x_samples


def simulator_torch_snpla_batched(theta):
    n_sim, data_dim = theta.shape
    x_samples = torch.zeros((n_sim, data_dim))

    for i in range(n_sim):
        x_samples[i, ] = simulator_torch(theta[i,])

    return x_samples.to(dtype=torch.float32)