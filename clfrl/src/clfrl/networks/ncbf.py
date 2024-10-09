from typing import Type

import flax.linen as nn
import jax.numpy as jnp
from flax.linen import initializers
from jaxtyping import Float

from clfrl.networks.network_utils import default_nn_init
from clfrl.utils.jax_types import Arr
from clfrl.utils.jax_utils import unmerge01, unmergelast


class SingleValueFn(nn.Module):
    net_cls: Type[nn.Module]

    @nn.compact
    def __call__(self, obs: Float[Arr, "* nobs"]) -> Float[Arr, "*"]:
        mlp = self.net_cls()
        x = mlp(obs)
        out = nn.Dense(1, kernel_init=default_nn_init())(x)
        out = out.squeeze(axis=-1)
        return out


class MultiValueFn(nn.Module):
    net_cls: Type[nn.Module]
    nout: int

    @nn.compact
    def __call__(self, obs: Float[Arr, "* nobs"]) -> Float[Arr, "*"]:
        mlp = self.net_cls()
        x = mlp(obs)
        return nn.Dense(self.nout, kernel_init=default_nn_init())(x)


class MultiNormValueFn(nn.Module):
    net_cls: Type[nn.Module]
    nout: int
    # Dimension of the latent space.
    n_vec: int

    @nn.compact
    def __call__(self, obs: Float[Arr, "* nobs"]) -> Float[Arr, "*"]:
        x = self.net_cls()(obs)
        x = nn.Dense(self.nout * self.n_vec, kernel_init=default_nn_init())(x)
        # (..., n_out * n_vec) -> (..., n_out, n_vec)
        hx_x = unmergelast(x, (self.nout, self.n_vec))
        # (n_out, n_vec) -> (n_out, )
        h_V = jnp.sum(hx_x * hx_x, axis=-1)
        # This is non-negative. We need to shift it so that it can be negative.
        h_V = h_V + self.param("shift", initializers.constant(-1.0), (1,))
        return h_V
