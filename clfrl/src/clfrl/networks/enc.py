import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import random

 
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

class ZWrapper(nn.Module):
    base_cls: nn.Module
    nz: int

    @nn.compact
    def __call__(self, state, z):
        z_encoded = BFloatEncoder(nz=self.nz)(z)
        concatenated = jnp.concatenate([state, z_encoded], axis=-1)
        output = self.base_cls()(concatenated)
        return output

# Example usage:
# key = random.PRNGKey(0)
# base_network = nn.Dense(256)  # Example base network
# encoder = StateZEncoder(base_cls=base_network, nz=10)
# state = jnp.array([0.5, 0.2, 0.1])
# z = jnp.array([1.0], dtype=jnp.bfloat16)
# params = encoder.init(key, state, z)
# output_vector = encoder.apply(params, state, z)
# print(output_vector)