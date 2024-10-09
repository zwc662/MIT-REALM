import functools as ft
from typing import Type

import flax.linen as nn
import jax.numpy as jnp
from jaxtyping import Float

from clfrl.dyn.dyn_types import Obs
from clfrl.networks.network_utils import default_nn_init
from clfrl.networks.fourier_emb import pos_embed_random
from clfrl.utils.jax_types import Arr


class NormSqValueFn(nn.Module):
    net_cls: Type[nn.Module]

    @nn.compact
    def __call__(self, obs: Float[Arr, "* nobs"]) -> Float[Arr, "*"]:
        x = self.net_cls()(obs)
        return jnp.sum(x * x, axis=-1)


class NormSqMetricFn(nn.Module):
    net_cls: Type[nn.Module]
    goal_obs: Obs

    @nn.compact
    def __call__(self, obs: Float[Arr, "* nobs"]) -> Float[Arr, "*"]:
        mlp = self.net_cls()
        x = mlp(obs)
        goal = mlp(self.goal_obs)
        dx = x - goal
        assert dx.shape == x.shape
        return jnp.sum(dx * dx, axis=-1)


class NormSqMetricFn2(nn.Module):
    net_cls: Type[nn.Module]
    goal_obs: Obs
    embed_dim: int = 32
    latent_dim: int = 32

    @nn.compact
    def __call__(self, obs: Float[Arr, "* nobs"]) -> Float[Arr, "*"]:
        mlp = self.net_cls()

        # Apply positional encoding.
        pos_emb = ft.partial(pos_embed_random, self.embed_dim // 2)
        # pos_emb = lambda x: x
        latent_emb = nn.Dense(self.latent_dim, kernel_init=default_nn_init())

        obs_emb, goal_emb = pos_emb(obs), pos_emb(self.goal_obs)

        x = latent_emb(mlp(obs_emb))
        goal = latent_emb(mlp(goal_emb))
        dx = x - goal
        assert dx.shape == x.shape
        return jnp.sum(dx * dx, axis=-1)
