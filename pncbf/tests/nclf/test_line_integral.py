import ipdb
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from diffrax import ConstantStepSize, ODETerm, PIDController, SaveAt, Tsit5, diffeqsolve

from pncbf.utils.jax_utils import jax_jit_np, jax_use_cpu, jax_use_double, jax_vmap
from pncbf.utils.path_utils import mkdir
from pncbf.utils.paths import get_script_plot_dir


def fn(x):
    return jnp.sin(4 * x) + jnp.cos(x) + 0.3 * x


def xdot(x):
    return jnp.sin(3 * x) + 1


def main():
    jax_use_cpu()
    jax_use_double()
    plot_dir = mkdir(get_script_plot_dir() / "line_integral")

    grad_fn = jax.grad(fn)

    # 0: Integrate xs.
    x0 = 0.0

    # dt = 0.15
    # T = 11

    dt = 0.05
    T = int(1.5 / dt) + 1

    T_t = np.arange(T) * dt

    term = ODETerm(lambda t, x, args: xdot(x))
    solver = Tsit5()
    saveat = SaveAt(ts=T_t)
    ss_control = PIDController(pcoeff=0.4, icoeff=1.0, rtol=1e-5, atol=1e-6)
    solution = diffeqsolve(term, solver, t0=0, t1=T_t[-1], dt0=dt, y0=x0, saveat=saveat, stepsize_controller=ss_control)
    T_x = np.array(solution.ys)

    fig, ax = plt.subplots(layout="constrained")
    ax.plot(T_t, T_x)
    fig.savefig(plot_dir / "xs.pdf")

    T_f = jax_jit_np(jax_vmap(fn))(T_x)

    # 1: Use property of line integral on conservative vector field.
    T_int_exact = T_f - T_f[0]

    # 2: Use trapezoidal rule.
    T_grad = jax_vmap(grad_fn)(T_x)
    T_xdot = np.gradient(T_x, T_t)
    T_grad_xdot = T_grad * T_xdot
    T_int_trap_np = [np.trapz(T_grad_xdot[:kk], T_t[:kk], axis=0) for kk in range(1, T + 1)]
    T_int_trap_jnp = [jnp.trapz(T_grad_xdot[:kk], T_t[:kk], axis=0) for kk in range(1, T + 1)]

    fig, axes = plt.subplots(2, layout="constrained")
    axes[0].plot(T_t, T_int_exact, label="Exact")
    axes[0].plot(T_t, T_int_trap_np, ls="--", label="np Trap")
    axes[0].plot(T_t, T_int_trap_jnp, ls="--", label="jnp Trap")
    axes[0].legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, 1.0), fontsize="x-small")

    axes[1].plot(T_t, T_int_exact - T_int_trap_np, label="Exact - np Trap")
    axes[1].set_title("Error")
    fig.savefig(plot_dir / "int.pdf")


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
