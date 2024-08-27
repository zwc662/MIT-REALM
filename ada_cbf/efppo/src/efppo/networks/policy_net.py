from typing import Type

import flax.linen as nn
import jax.numpy as jnp

from efppo.networks.network_utils import default_nn_init
from efppo.task.dyn_types import Obs
from efppo.utils.tfp import tfd


class DiscretePolicyNet(nn.Module):
    base_cls: Type[nn.Module]
    n_actions: int
    scale: float = 3.0

    @nn.compact
    def __call__(self, obs: Obs, *args, **kwargs) -> tfd.Distribution:
        x = self.base_cls()(obs, *args, **kwargs)
        logits = nn.Dense(self.n_actions, kernel_init=default_nn_init(), name="OutputDense")(x)
        # Output the mean and log-standard deviation for the 2D Gaussian
        mean = nn.Dense(2, kernel_init=nn.initializers.xavier_uniform())(x)
        log_std = nn.Dense(2, kernel_init=nn.initializers.xavier_uniform())(x)
        std = jnp.exp(log_std)
        # Constrain the mean to be within (0, 3) using sigmoid scaling
        mean = self.scale * nn.sigmoid(mean)

        # Construct the 2D Multivariate Normal distribution
        dist = tfd.MultivariateNormalDiag(loc=mean, scale_diag=std)

        return dist
        #return tfd.Categorical(logits=logits)
