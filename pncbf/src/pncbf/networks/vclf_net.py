from typing import Type

import flax.linen as nn
from jaxtyping import Float

from pncbf.networks.network_utils import default_nn_init
from pncbf.utils.jax_types import Arr


class VecFn(nn.Module):
    net_cls: Type[nn.Module]
    out_dim: int

    @nn.compact
    def __call__(self, obs: Float[Arr, "* nobs"]) -> Float[Arr, "*"]:
        x = self.net_cls()(obs)
        x = nn.Dense(self.out_dim, kernel_init=default_nn_init())(x)
        return x
