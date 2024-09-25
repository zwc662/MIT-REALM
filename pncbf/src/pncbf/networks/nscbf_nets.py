from typing import Type

import flax.linen as nn
import jax.numpy as jnp
from flax.linen import initializers
from jaxtyping import Float

from pncbf.dyn.dyn_types import Obs, State
from pncbf.networks.network_utils import default_nn_init
from pncbf.utils.jax_types import Arr
from pncbf.utils.jax_utils import unmerge01, unmergelast


class NSCBFValueFn(nn.Module):
    net_cls: Type[nn.Module]
    obs_eq: Obs

    @nn.compact
    def __call__(self, obs: Float[Arr, "* nobs"]) -> Float[Arr, "*"]:
        mlp = self.net_cls()
        out_dense = nn.Dense(1, kernel_init=default_nn_init())

        out = out_dense(mlp(obs)).squeeze(axis=-1)
        out_eq = out_dense(mlp(self.obs_eq)).squeeze(axis=-1)

        return (out - out_eq) ** 2
