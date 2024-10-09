from typing import Type

import flax.linen as nn
import jax.numpy as jnp

from clfrl.dyn.dyn_types import Obs
from clfrl.networks.network_utils import default_nn_init
from clfrl.utils.tfp import TanhTransformedDistribution, tfd


class PolDet(nn.Module):
    base_cls: Type[nn.Module]
    _nu: int

    @nn.compact
    def __call__(self, obs: Obs, *args, **kwargs) -> tfd.Distribution:
        x = self.base_cls()(obs, *args, **kwargs)
        out = nn.Dense(self.nu, kernel_init=default_nn_init(), name="Output")(x)
        out = jnp.tanh(out)
        return out

    @property
    def nu(self):
        return self._nu
