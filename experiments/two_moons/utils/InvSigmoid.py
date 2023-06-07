# inverse sigmoid transformation

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
        # if torch.min(inputs) <= -1 or torch.max(inputs) >= 1:
        #    raise InputOutsideDomain()
        outputs = 0.5 * torch.log((1 + inputs) / (1 - inputs))
        logabsdet = -torch.log(1 - inputs ** 2)
        logabsdet = torchutils.sum_except_batch(logabsdet, num_batch_dims=1)
        return outputs, logabsdet

class InvSigmoid(Transform):
    def __init__(self, temperature=1, eps=1e-6, learn_temperature=False):
        super().__init__()
        self.eps = eps
        if learn_temperature:
            self.temperature = nn.Parameter(torch.tensor([temperature]))
        else:
            self.temperature = torch.tensor([temperature])

    def forward(self, inputs, context=None):
        #if torch.min(inputs) < 0 or torch.max(inputs) > 1:
        #    raise InputOutsideDomain()
        temperature = self.temperature.to(inputs.device)
        inputs = torch.clamp(inputs, self.eps, 1 - self.eps)

        outputs = (1 / temperature) * (torch.log(inputs) - torch.log1p(-inputs))
        logabsdet = -torchutils.sum_except_batch(
            torch.log(temperature)
            - F.softplus(-temperature * outputs)
            - F.softplus(temperature * outputs)
        )
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        temperature = self.temperature.to(inputs.device)
        inputs = temperature * inputs
        outputs = torch.sigmoid(inputs)
        logabsdet = torchutils.sum_except_batch(
            torch.log(temperature) - F.softplus(-inputs) - F.softplus(inputs)
        )
        return outputs, logabsdet
