from typing import Type

import flax.linen as nn
from jaxtyping import Float, Int

from efppo.networks.network_utils import default_nn_init
from efppo.utils.jax_types import Arr
from efppo.utils.shape_utils import assert_shape



class DiscreteCriticNet(nn.Module):
    net_cls: Type[nn.Module]
    n_actions: int

    @nn.compact
    def __call__(self, state: Float[Arr, "* nx"], *args, **kwargs) -> Float[Arr, "*"]:
        batch_shape = state.shape[:-1]
        x = self.net_cls()(state, *args, **kwargs)
        critic = nn.Dense(self.n_actions, kernel_init=default_nn_init())(x) 
        return assert_shape(critic, batch_shape + (self.n_actions,))
 
class DoubleDiscreteCriticNet(nn.Module):
    net_cls: Type[nn.Module]
    n_actions: int
    n_critics: int = 2

    @nn.compact
    def __call__(self, state: Float[Arr, "* nx"], *args, **kwargs) -> Float[Arr, "*"]:
        batch_shape = (*state.shape[:-1], self.n_actions)
        double_net = nn.vmap(
            DiscreteCriticNet, 
            variable_axes = {'params': 0},
            split_rngs = {'params': True},
            in_axes = None,
            out_axes = 0,
            axis_size = self.n_critics)
        return double_net(self.net_cls, self.n_actions)(state, *args, **kwargs)
       
