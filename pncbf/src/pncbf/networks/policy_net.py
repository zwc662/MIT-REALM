from typing import Type

import flax.linen as nn
import jax.numpy as jnp

from pncbf.dyn.dyn_types import Obs
from pncbf.networks.network_utils import default_nn_init
from pncbf.utils.tfp import TanhTransformedDistribution, tfd


class TanhNormal(nn.Module):
    base_cls: Type[nn.Module]
    _nu: int
    log_std_min: float = -20
    log_std_max: float = 2

    @nn.compact
    def __call__(self, obs: Obs, *args, **kwargs) -> tfd.Distribution:
        x = self.base_cls()(obs, *args, **kwargs)
        means = nn.Dense(self.nu, kernel_init=default_nn_init(), name="OutputDenseMean")(x)
        log_stds = nn.Dense(self.nu, kernel_init=default_nn_init(), name="OutputDenseLogStd")(x)
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)
        distribution = tfd.Normal(loc=means, scale=jnp.exp(log_stds))
        return tfd.Independent(TanhTransformedDistribution(distribution), reinterpreted_batch_ndims=1)

    @property
    def nu(self):
        return self._nu
