from typing import Union

import torch
from torch import distributions

from nflows.distributions.base import Distribution
from nflows.utils import torchutils


class UniformContext(Distribution):
    def __init__(
            self,
            low: Union[torch.Tensor, float],
            high: Union[torch.Tensor, float],
            shape
    ):
        super().__init__()
        self.dist = distributions.Uniform(low=low, high=high)
        self._shape = torch.Size(shape)

    def _log_prob(self, inputs, context):
        # Note: the context is ignored.
        if inputs.shape[1:] != self._shape:
            raise ValueError(
                "Expected input of shape {}, got {}".format(
                    self._shape, inputs.shape[1:]
                )
            )
        return self.dist.log_prob(inputs).sum(dim=1)

    def _sample(self, num_samples, context):
        if context is None:
            return self.dist.rsample(sample_shape=[num_samples]) #torch.randn(num_samples, *self._shape, device=self._log_z.device)
        else:
            # The value of the context is ignored, only its size and device are taken into account.
            context_size = context.shape[0]
            # samples = self.dist.low + (self.dist.high - self.dist.low) * torch.rand(context_size * num_samples,
            #                                                                        *self._shape,
            #                                                                        device=context.device)

            # context_size * num_samples is used to adjust for how many cases of context we have!
            samples = self.dist.rsample(sample_shape=[context_size * num_samples])

            return torchutils.split_leading_dim(samples, [context_size, num_samples])
