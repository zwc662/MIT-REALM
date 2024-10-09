import jax
import jax.numpy as jnp
import jax.tree_util as jtu


# Copied from the official SAM GitHub repository. Note how it doesn't add an
# epsilon to the gradient norm before normalizing the gradients.
def dual_vector(y: jnp.ndarray) -> jnp.ndarray:
    """Returns the solution of max_x y^T x s.t.

    ||x||_2 <= 1.

    Args:
      y: A pytree of numpy ndarray, vector y in the equation above.
    """
    gradient_norm = jnp.sqrt(sum([jnp.sum(jnp.square(e)) for e in jtu.tree_leaves(y)]))
    normalized_gradient = jtu.tree_map(lambda x: x / gradient_norm, y)
    return normalized_gradient


def sharpness_aware_minimization(rho: float, loss_fn, params):
    updates, info = jax.grad(loss_fn, has_aux=True)(params)
    updates = dual_vector(updates)
    noised_params = jtu.tree_map(lambda p: p + rho * updates, params)
    updates_noised, info_noised = jax.grad(loss_fn, has_aux=True)(noised_params)

    return updates, updates_noised, info, info_noised
