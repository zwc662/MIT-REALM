from typing import Generic, Type, TypeVar

import flax.linen as nn
import jax.numpy as jnp
from flax.linen import initializers
from jaxtyping import Float

from pncbf.networks.network_utils import default_nn_init
from pncbf.utils.jax_types import Arr

 
_WrappedModule = TypeVar("_WrappedModule", bound=nn.Module)


class ZEncoder(nn.Module):
    nz: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(self.nz)(x)
        return x

# Example usage:
# key = random.PRNGKey(0)
# encoder = BFloatEncoder(nz=10)
# input_tensor = jnp.array([1.0], dtype=jnp.bfloat16)
# params = encoder.init(key, input_tensor)
# output_vector = encoder.apply(params, input_tensor)
# print(output_vector)

class ZWrapper(nn.Module, Generic[_WrappedModule]):
    """Wrapper for networks that only take in the state to also take in z."""

    base_cls: Type[_WrappedModule]
    z_encoder_cls: Type[nn.Module]

    @nn.compact
    def __call__(self, obs: Obs, z: FloatScalar, *args, **kwargs) -> AnyFloat:
        assert obs.ndim == (z.ndim + 1)
        z = z[..., None]
        enc_z = self.z_encoder_cls()(z)
        feat = jnp.concatenate([obs, enc_z, *args, *list(kwargs.values())], axis=-1)
        return self.base_cls()(feat)

# Example usage:
# key = random.PRNGKey(0)
# base_network = nn.Dense(256)  # Example base network
# encoder = StateZEncoder(base_cls=base_network, nz=10)
# state = jnp.array([0.5, 0.2, 0.1])
# z = jnp.array([1.0], dtype=jnp.bfloat16)
# params = encoder.init(key, state, z)
# output_vector = encoder.apply(params, state, z)
# print(output_vector)