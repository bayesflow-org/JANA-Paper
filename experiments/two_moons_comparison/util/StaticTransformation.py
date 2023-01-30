# inverse sigmoid transformation

from nflows.transforms.base import (
    InputOutsideDomain,
    Transform,
)
import torch.nn as nn
from nflows.utils import torchutils
from torch.nn import functional as F
import torch


class InvSigmoidStatic(Transform):
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
        #logabsdet = -torchutils.sum_except_batch(
        #    torch.log(self.temperature)
        #    - F.softplus(-self.temperature * outputs)
        #    - F.softplus(self.temperature * outputs)
        #)
        batch_size = inputs.size(0)
        logabsdet = inputs.new_zeros(batch_size)
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        inputs = self.temperature * inputs
        outputs = torch.sigmoid(inputs)
        #logabsdet = torchutils.sum_except_batch(
        #    torch.log(self.temperature) - F.softplus(-inputs) - F.softplus(inputs)
        #)
        batch_size = inputs.size(0)
        logabsdet = inputs.new_zeros(batch_size)
        return outputs, logabsdet


from typing import Iterable, Optional, Tuple, Union
import torch
from torch import Tensor

from nflows.transforms.base import Transform

class PointwiseAffineTransformStatic(Transform):
    """Forward transform X = X * scale + shift."""

    def __init__(
        self, shift: Union[Tensor, float] = 0.0, scale: Union[Tensor, float] = 1.0,
    ):
        super().__init__()
        shift, scale = map(torch.as_tensor, (shift, scale))

        if not (scale > 0.0).all():
            raise ValueError("Scale must be strictly positive.")

        self.register_buffer("_shift", shift)
        self.register_buffer("_scale", scale)

    @property
    def _log_scale(self) -> Tensor:
        return torch.log(self._scale)

    # XXX Memoize result on first run?
    def _batch_logabsdet(self, batch_shape: Iterable[int]) -> Tensor:
        """Return log abs det with input batch shape."""

        if self._log_scale.numel() > 1:
            return self._log_scale.expand(batch_shape).sum()
        else:
            # When log_scale is a scalar, we use n*log_scale, which is more
            # numerically accurate than \sum_1^n log_scale.
            return self._log_scale * torch.Size(batch_shape).numel()

    def forward(self, inputs: Tensor, context=Optional[Tensor]) -> Tuple[Tensor]:
        batch_size, *batch_shape = inputs.size()

        # RuntimeError here means shift/scale not broadcastable to input.
        outputs = inputs * self._scale + self._shift
        #logabsdet = self._batch_logabsdet(batch_shape).expand(batch_size)
        batch_size = inputs.size(0)
        logabsdet = inputs.new_zeros(batch_size)

        return outputs, logabsdet

    def inverse(self, inputs: Tensor, context=Optional[Tensor]) -> Tuple[Tensor]:
        batch_size, *batch_shape = inputs.size()
        outputs = (inputs - self._shift) / self._scale
        #logabsdet = -self._batch_logabsdet(batch_shape).expand(batch_size)
        batch_size = inputs.size(0)
        logabsdet = inputs.new_zeros(batch_size)

        return outputs, logabsdet
