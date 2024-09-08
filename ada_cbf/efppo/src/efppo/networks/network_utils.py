from typing import Any, Callable, Literal, Sequence

import flax.linen as nn

import jax
import jax.numpy as jnp
import jax.random as jrd

import optax
from flax import traverse_util

from efppo.utils.jax_types import AnyFloat, FloatScalar, Shape
from efppo.utils.rng import PRNGKey

ActFn = Callable[[AnyFloat], AnyFloat]

InitFn = Callable[[PRNGKey, Shape, Any], Any]

default_nn_init = nn.initializers.xavier_uniform

HidSizes = Sequence[int]


def scaled_init(initializer: nn.initializers.Initializer, scale: float) -> nn.initializers.Initializer:
    def scaled_init_inner(*args, **kwargs) -> AnyFloat:
        return scale * initializer(*args, **kwargs)

    return scaled_init_inner


ActLiteral = Literal["relu", "tanh", "elu", "swish", "silu", "gelu", "softplus"]


def get_act_from_str(act_str: ActLiteral) -> ActFn:
    act_dict: dict[Literal, ActFn] = dict(
        relu=nn.relu, tanh=nn.tanh, elu=nn.elu, swish=nn.swish, silu=nn.silu, gelu=nn.gelu, softplus=nn.softplus
    )
    return act_dict[act_str]


def wd_mask(params):
    Path = tuple[str, ...]
    flat_params: dict[Path, jnp.ndarray] = traverse_util.flatten_dict(params)
    # Apply weight decay to all parameters except biases and LayerNorm scale and bias.
    flat_mask = {path: (path[-1] != "bias" and path[-2:] != ("LayerNorm", "scale")) for path in flat_params}
    return traverse_util.unflatten_dict(flat_mask)


def optim(learning_rate: float, wd: float, eps: float):
    opt = optax.adamw(learning_rate, eps=eps, weight_decay=wd, mask=wd_mask)
    opt = optax.apply_if_finite(opt, 100)
    return opt


def get_default_tx(
    lr: optax.Schedule | FloatScalar, wd: optax.Schedule | FloatScalar = 1e-4, eps: FloatScalar = 1e-5
) -> optax.GradientTransformation:
    return optax.inject_hyperparams(optim)(learning_rate=lr, wd=wd, eps=eps)

   
        
def rsample(logits, key, tau = 0.5):
    """Differentiable sampling from a categorical distribution using Gumbel-Softmax."""
    u = jrd.uniform(key, shape=logits.shape, minval=1e-6, maxval=1.0)
    gumbel_noise = -jnp.log(-jnp.log(u))
    y = logits + gumbel_noise
    softmax_output = jax.nn.softmax(y / tau, axis=-1)
    # Create a hard sample by taking the argmax, but keep gradients from soft_sample
    hardmax_output = jax.lax.stop_gradient(jax.nn.one_hot(jnp.argmax(softmax_output, axis=-1), num_classes=logits.shape[-1]))
    
    # Return the hard sample for the forward pass, but allow gradients from soft_sample
    sample = hardmax_output + jax.lax.stop_gradient(softmax_output - hardmax_output)

    # Compute log-softmax to get log-probabilities of the softmax distribution
    log_probabilities = jax.nn.log_softmax(logits, axis=-1)
    
    # Select the log-probability corresponding to the sampled category
    log_prob = jnp.sum(log_probabilities * sample, axis=-1)
    
    return sample, log_prob
