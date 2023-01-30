# functions for all methods
import torch
from nflows.flows.base import Flow
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.distributions.normal import StandardNormal
from nflows.transforms.permutations import ReversePermutation
from nflows.transforms.standard import PointwiseAffineTransform
from .simulation import TwoMoons

from nflows.transforms.base import (
    CompositeTransform,
)

# load from util (from https://stackoverflow.com/questions/714063/importing-modules-from-parent-folder)
import os, sys, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


from nflows.transforms.base import (
    InputOutsideDomain,
    Transform,
)
import torch.nn as nn
from nflows.utils import torchutils
from torch.nn import functional as F
import torch

class InvTanh(Transform):
    def inverse(self, inputs, context=None):
        outputs = torch.tanh(inputs)
        logabsdet = torch.log(1 - outputs ** 2)
        logabsdet = torchutils.sum_except_batch(logabsdet, num_batch_dims=1)
        return outputs, logabsdet

    def forward(self, inputs, context=None):
        if torch.min(inputs) <= -1 or torch.max(inputs) >= 1:
            raise InputOutsideDomain()
        outputs = 0.5 * torch.log((1 + inputs) / (1 - inputs))
        logabsdet = -torch.log(1 - inputs ** 2)
        logabsdet = torchutils.sum_except_batch(logabsdet, num_batch_dims=1)
        return outputs, logabsdet

class InvSigmoid(Transform):
    def __init__(self, temperature=1, eps=1e-6, learn_temperature=False):
        super().__init__()
        self.eps = eps
        if learn_temperature:
            self.temperature = nn.Parameter(torch.Tensor([temperature]))
        else:
            self.temperature = torch.Tensor([temperature])

    def forward(self, inputs, context=None):
        if torch.min(inputs) < 0 or torch.max(inputs) > 1:
            raise InputOutsideDomain()

        inputs = torch.clamp(inputs, self.eps, 1 - self.eps)

        outputs = (1 / self.temperature) * (torch.log(inputs) - torch.log1p(-inputs))
        logabsdet = -torchutils.sum_except_batch(
            torch.log(self.temperature)
            - F.softplus(-self.temperature * outputs)
            - F.softplus(self.temperature * outputs)
        )
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        inputs = self.temperature * inputs
        outputs = torch.sigmoid(inputs)
        logabsdet = torchutils.sum_except_batch(
            torch.log(self.temperature) - F.softplus(-inputs) - F.softplus(inputs)
        )
        return outputs, logabsdet



# sets up the models
def set_up_model(prior, mean_radius=1.0, sd_radius=0.1, baseoffset=1.0, fixed_data=True):
    model = TwoMoons(mean_radius, sd_radius, baseoffset, fixed_data)

    if fixed_data:
        x_o = torch.zeros(2)

    return x_o, model


# sets up the networks for the flow and likelihood and posterior model
def set_up_networks(seed=10, dim=2):
    torch.manual_seed(seed)
    base_dist_lik = StandardNormal(shape=[2])

    num_layers = 5

    transforms = []
    for _ in range(num_layers):
        transforms.append(ReversePermutation(features=2))
        transforms.append(MaskedAffineAutoregressiveTransform(features=2,
                                                              hidden_features=50,
                                                              context_features=dim,
                                                              num_blocks=1))

    transform = CompositeTransform(transforms)

    flow_lik = Flow(transform, base_dist_lik)

    base_dist_post = StandardNormal(
        shape=[dim])  # BoxUniform(low=-2*torch.ones(2), high=2*torch.ones(2)) #StandardNormal(shape=[dim])

    # base_dist_post = BoxUniform(low=-2*torch.ones(2), high=2*torch.ones(2))

    num_layers = 5

    transforms = []

    num_off_set = 0.0001  # numerical offset since the prior is on the open space
    #shift, scale = calc_scale_and_shift(-1, 1)

    #print(shift)
    #print(scale)

    transforms.append(PointwiseAffineTransform(shift=0.5, scale=1 / 4.0))
    #transforms.append(PointwiseAffineTransform(shift=shift, scale=scale))

    transforms.append(InvSigmoid.InvSigmoid())  # this should be inv sigmoide!

    for _ in range(num_layers):
        transforms.append(ReversePermutation(features=dim))
        transforms.append(MaskedAffineAutoregressiveTransform(features=dim,
                                                              hidden_features=50,
                                                              context_features=2,
                                                              num_blocks=1))

    transform = CompositeTransform(transforms)

    flow_post = Flow(transform, base_dist_post)

    return flow_lik, flow_post


def calc_scale_and_shift(lower, upper):
    sigma_lower = 0
    sigma_upper = 1

    scale = (sigma_upper - sigma_lower) / (upper - lower)
    shift = sigma_lower - lower * scale

    return shift, scale


def sample_hp(method, case):
    torch.manual_seed(case)

    if method == "snpe_c" or method == "snl" or method == "snre_b":
        return 10 ** -4 + (10 ** -2 - 10 ** -4) * torch.rand(1)
    else:
        lr_like = 10 ** -4 + (10 ** -2 - 10 ** -4) * torch.rand(1)
        lr_post = 10 ** -4 + (10 ** -2 - 10 ** -4) * torch.rand(1)
        gamma_post = 0.8 + (0.999 - 0.8) * torch.rand(1)
        lam = 0.65 + (0.95 - 0.65) * torch.rand(1)
        return [lr_like[0].item(), lr_post[0].item(), gamma_post[0].item(), lam[0].item()]
