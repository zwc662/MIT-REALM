import jax
import jax.numpy as jnp
import jax.vmap as vmap

# Example kernel function: RBF Kernel
def rbf_kernel(x, y, h=1.0):
    """Radial Basis Function (RBF) kernel."""
    diff = x - y
    dist_sq = jnp.sum(diff**2)
    return jnp.exp(-dist_sq / (2 * h**2))

def compute_kernel_matrix(params):
    """Compute the kernel matrix for a list of parameter sets."""
    kernel_matrix = vmap(lambda p_i: vmap(lambda p_j: rbf_kernel(p_i, p_j))(params))(params)
    return kernel_matrix

def compute_kernel_gradient(params):
    """Compute the gradient of the kernel matrix with respect to the parameters."""
    def single_kernel_grad(p_i, p_j):
        kernel_value = rbf_kernel(p_i, p_j)
        grad_kernel_value = jax.grad(lambda p: rbf_kernel(p, p_j))(p_i)
        return grad_kernel_value * kernel_value
    
    kernel_gradients = vmap(lambda p_i: vmap(lambda p_j: single_kernel_grad(p_i, p_j))(params))(params)
    return kernel_gradients

def svgd_update(params, grads, kernel_matrix, kernel_gradients):
    """Compute the SVGD update for each parameter set in the ensemble."""
    K = len(params)
    updates = []
    for i in range(K):
        update = jnp.mean(kernel_matrix[i, :, None] * grads + kernel_gradients[i], axis=0)
        updates.append(update)
    return updates