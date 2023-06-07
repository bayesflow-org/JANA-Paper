import torch
from nflows.flows.base import Flow
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.distributions.normal import StandardNormal
from nflows.transforms.permutations import ReversePermutation
from nflows.transforms.standard import PointwiseAffineTransform

from nflows.transforms.base import CompositeTransform

from utils.InvSigmoid import InvSigmoid


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

    base_dist_post = StandardNormal(shape=[dim])

    num_layers = 5

    transforms = []

    num_off_set = 0.0001  # numerical offset since the prior is on the open space

    transforms.append(PointwiseAffineTransform(shift=0.5, scale=1 / 4.0))

    transforms.append(InvSigmoid())

    for _ in range(num_layers):
        transforms.append(ReversePermutation(features=dim))
        transforms.append(MaskedAffineAutoregressiveTransform(features=dim,
                                                              hidden_features=50,
                                                              context_features=2,
                                                              num_blocks=1))

    transform = CompositeTransform(transforms)

    flow_post = Flow(transform, base_dist_post)

    return flow_lik, flow_post
