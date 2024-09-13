from typing import Generic, Type, TypeVar

import flax.linen as nn
import jax.numpy as jnp
from jaxtyping import Float

from efppo.networks.network_utils import default_nn_init
from efppo.task.dyn_types import Obs, Control
from efppo.utils.jax_types import AnyFloat, BFloat, FloatScalar
from efppo.utils.shape_utils import assert_shape
from efppo.utils.jax_types import Arr
from efppo.utils.tfp import tfd


_WrappedModule = TypeVar("_WrappedModule", bound=nn.Module)


class ZEncoder(nn.Module):
    nz: int
    z_mean: float
    z_scale: float

    @nn.compact
    def __call__(self, z: BFloat) -> BFloat:
        # 1: Normalize z.
        norm_z = (z - self.z_mean) / self.z_scale

        # 2: Encode it.
        enc_z = nn.Dense(self.nz, kernel_init=default_nn_init())(norm_z)
        enc_z = nn.tanh(enc_z)
        return assert_shape(enc_z, self.nz, "enc_z")


class EFWrapper(nn.Module, Generic[_WrappedModule]):
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
 
class BoltzmanPolicyWrapper(nn.Module, Generic[_WrappedModule]):
    net: _WrappedModule
   
    def __call__(self, params, *args, **kwargs) -> tfd.Distribution:
        qs = self.net.apply(params, *args, **kwargs)
        ## The critics output the cost, not the reward
        ## The lower the cost is, the higher the probability is
        exp_qs = jnp.exp(-qs)
        p = exp_qs / exp_qs.sum(axis = -1, keepdims=False)[..., jnp.newaxis]
        logits = jnp.log(p)
        return tfd.Categorical(logits=logits)
    
class EnsembleBoltzmanPolicyWrapper(nn.Module, Generic[_WrappedModule]):
    ensemble_net: _WrappedModule

    def __call__(self, params, *args, **kwargs) -> tfd.Distribution:
        qs_all = self.ensemble_net.apply(params, *args, **kwargs)
        qs = qs_all.mean(axis = -2, keepdims = False)
        exp_qs = jnp.exp(-qs)
        p = exp_qs / exp_qs.sum(axis = -1, keepdims=False)[..., jnp.newaxis]
        logits = jnp.log(p)
        
        return tfd.Categorical(logits=logits)