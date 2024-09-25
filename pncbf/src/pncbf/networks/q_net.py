from typing import Type

import flax.linen as nn
import jax.numpy as jnp
from flax.linen import initializers
from jaxtyping import Float

from pncbf.networks.network_utils import default_nn_init
from pncbf.utils.jax_types import Arr
from pncbf.utils.shape_utils import assert_shape


class QValueNet(nn.Module):
    net_cls: Type[nn.Module]
    n_out: int

    @nn.compact
    def __call__(self, obs: Float[Arr, "* nobs"], control: Float[Arr, "* nu"], *args, **kwargs) -> Float[Arr, "*"]:
        batch_shape = obs.shape[:-1]
        inputs = jnp.concatenate([obs, control], axis=-1)
        x = self.net_cls()(inputs, *args, **kwargs)
        Ql = nn.Dense(self.n_out, kernel_init=default_nn_init())(x)
        return assert_shape(Ql, (*batch_shape, self.n_out))


class QValueRobNet(nn.Module):
    net_cls: Type[nn.Module]
    n_out: int

    @nn.compact
    def __call__(
        self, obs: Float[Arr, "* nobs"], control: Float[Arr, "* nu"], disturb: Float[Arr, "* nd"], *args, **kwargs
    ) -> Float[Arr, "*"]:
        batch_shape = obs.shape[:-1]
        inputs = jnp.concatenate([obs, control, disturb], axis=-1)
        x = self.net_cls()(inputs, *args, **kwargs)
        Ql = nn.Dense(self.n_out, kernel_init=default_nn_init())(x)
        return assert_shape(Ql, (*batch_shape, self.n_out))
