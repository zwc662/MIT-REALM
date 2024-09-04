import jax
import jax.numpy as jnp
import jax.tree_util as jtu

# Example kernel function: RBF Kernel
def rbf_kernel(x, y, h=1.0):
    """Radial Basis Function (RBF) kernel."""
    diff = x - y
    dist_sq = jnp.sum(diff**2)
    return jnp.exp(-dist_sq / (2 * h**2))

def flatten_params(params):
    """Flatten the parameter dictionary to a single array."""
    flat_params, _ = jax.flatten_util.ravel_pytree(params)
    return flat_params 

def unflatten_params(target_params, params):
    _, unravel_fn = jax.flatten_util.ravel_pytree(target_params)
    unflattened_params = unravel_fn(params)
    return unflattened_params

def compute_kernel_matrix(params):
    """Compute the kernel matrix for a list of parameter sets."""
    # Flatten the parameter dictionaries
    flat_params = jax.vmap(flatten_params)(params)
    
    # Compute the kernel matrix using the flattened parameters
    kernel_matrix = jax.vmap(lambda p_i: jax.vmap(lambda p_j: rbf_kernel(p_i, p_j))(flat_params))(flat_params)
    return kernel_matrix

def compute_kernel_gradient(params):
    """Compute the gradient of the kernel matrix with respect to the parameters."""
    def single_kernel_grad(p_i, p_j):
        kernel_value = rbf_kernel(p_i, p_j)
        grad_kernel_value = jax.grad(lambda p: rbf_kernel(p, p_j))(p_i)
        return grad_kernel_value * kernel_value
    flat_params = jax.vmap(flatten_params)(params)
    kernel_gradients = jax.vmap(lambda p_i: jax.vmap(lambda p_j: single_kernel_grad(p_i, p_j))(flat_params))(flat_params)
    return kernel_gradients

def svgd_update(grads, kernel_matrix, kernel_gradients):
    """Compute the SVGD update for each parameter set in the ensemble."""

    # Flatten the parameters
    flat_grads = jax.vmap(flatten_params)(grads)
     
    updates = jax.vmap(lambda k_mat, k_grad, grad: (k_mat[:, None] * grad + k_grad).mean(axis = 0), in_axes = 0)(
        kernel_matrix,
        kernel_gradients,
        flat_grads
    )

    # Convert updates back to the original parameter structure
    new_grads = jax.vmap(unflatten_params)(grads, updates)

    return new_grads