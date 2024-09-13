from typing import Type, List, Tuple

import flax.linen as nn
from jaxtyping import Float, Int
import jax.numpy as jnp
import jax
import jax.random as jrd

from efppo.networks.network_utils import default_nn_init
from efppo.task.dyn_types import Obs
from efppo.utils.tfp import tfd
from efppo.utils.jax_types import Arr

class ContinuousPolicyNet(nn.Module):
    base_cls: Type[nn.Module]
    output_size: int
    
    @nn.compact
    def __call__(self, obs: Obs, *args, **kwargs) -> tfd.Distribution:
        x = self.base_cls()(obs, *args, **kwargs)
        # Output the mean and log-standard deviation for the 2D Gaussian
        mean_dense = nn.Dense(self.output_size, kernel_init=default_nn_init(), name="MeanDense")(x)
        mean = nn.tanh(mean_dense)
        
        log_std = nn.Dense(self.output_size, kernel_init=default_nn_init(), name="LogStd")(x)
        std = jnp.exp(log_std)
        # Constrain the mean to be within (0, 3) using sigmoid scaling
         
        # Construct the 2D Multivariate Normal distribution
        dist = tfd.MultivariateNormalDiag(loc=mean, scale_diag=std)

        return dist
      
class DiscretePolicyNet(nn.Module):
    base_cls: Type[nn.Module]
    n_actions: int
 
    @nn.compact
    def __call__(self, obs: Obs, *args, **kwargs) -> tfd.Distribution:
        x = self.base_cls()(obs, *args, **kwargs)
        logits = nn.Dense(self.n_actions, kernel_init=default_nn_init(), name="OutputDense")(x)
        return tfd.Categorical(logits=logits)
    
    def get_logits(self, obs: Obs, *args, **kwargs) -> Float[Arr, '*']:
        dist = self.__call__(obs, *args, **kwargs)
        return dist.logits
   

class EnsembleDiscretePolicyNet(nn.Module):
    base_cls: Type[nn.Module]
    n_actions: int
    n_policies: int = 2

    @nn.compact
    def __call__(self, obs: Obs, *args, **kwargs) -> Float[Arr, "*"]:
        ensemble_net = nn.vmap(
            DiscretePolicyNet, 
            variable_axes = {'params': 0},
            split_rngs = {'params': True},
            in_axes = None,
            out_axes = 0,
            axis_size = self.n_policies,
            methods = ('get_logits', 'rsample')
            )
        return ensemble_net(self.base_cls, self.n_actions).get_logits(obs, *args, **kwargs)

    def ensembble_dist(self, obs: Obs, *args, **kwargs) -> Float[Arr, "*"]:
        logits_all = self.__call__(obs, *args, **kwargs)
        logits = logits_all.mean(axis=-2, keepdims=False)
        return tfd.Categorical(logits=logits)
       
    