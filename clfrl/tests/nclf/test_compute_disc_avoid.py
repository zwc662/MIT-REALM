import ipdb
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from diffrax import ODETerm, SaveAt, Tsit5, diffeqsolve
from loguru import logger

from clfrl.ncbf.compute_disc_avoid import compute_all_disc_avoid_terms, compute_disc_avoid_terms, compute_disc_int_trap
from clfrl.utils.jax_utils import jax2np, jax_use_cpu, jax_use_double, jax_vmap
from clfrl.utils.paths import get_script_plot_dir


def h(t):
    c = 0.1337
    return c * t


def main():
    jax_use_cpu()
    jax_use_double()

    dt = 0.1
    lam = 0.29
    T = 50
    T_t = np.arange(T) * dt
    T_h = jax2np(h(T_t))
    Th_h = T_h[:, None]
    assert Th_h.shape == (T, 1)

    VT = 0.0

    terms = compute_disc_avoid_terms(lam, dt, Th_h)
    Vh_0_discr = np.maximum(terms.h_max_lhs, terms.discount * VT)

    Th_V, Th_disc_int, T_gammas = compute_all_disc_avoid_terms(lam, dt, Th_h)

    # Make sure compute_one and the compute_all give the same result at x_0.
    assert np.allclose(terms.discount, T_gammas[0])
    assert np.allclose(terms.h_max_lhs, Th_V[0])
    assert np.allclose(terms.h_disc_int_rhs, Th_disc_int[0])

    # Solve int using trapezoidal.
    Th_disc_int_trap = compute_disc_int_trap(lam, dt, Th_h)

    # Solve using diffrax.
    def f(t, y, args):
        dy = lam * jnp.exp(-lam * t) * jnp.array([h(t)])
        return dy

    term = ODETerm(f)
    solver = Tsit5()
    y0 = np.array([0.0])
    saveat = SaveAt(dense=True)
    solution = diffeqsolve(term, solver, t0=0.0, t1=T * dt, dt0=0.1 * dt, y0=y0, saveat=saveat)

    @jax.jit
    def solve(t0):
        def f_new(t, y, args):
            dy = lam * jnp.exp(-lam * (t - t0)) * jnp.array([h(t)])
            return dy

        term_new = ODETerm(f_new)

        saveat = SaveAt(dense=True)
        sol = diffeqsolve(term_new, solver, t0=t0, t1=T * dt, dt0=0.1 * dt, y0=y0, saveat=saveat)
        return sol
        # integral = sol.ys[-1]
        # assert integral.shape == (1,)
        # return integral

    T_int_cts = jax2np(jax.vmap(solution.evaluate)(T_t))

    logger.info("Computing V maunally...")
    T_V_cts = []
    for ii in range(T):
        t0 = T_t[ii]
        T_delta_ts = T_t[ii:] - t0
        sol = solve(t0)
        T_integral = jax_vmap(sol.evaluate)(T_t[ii:]).flatten()
        # T_integral = .evaluate()
        T_lhs = T_integral + np.exp(-lam * T_delta_ts) * T_h[ii:]
        assert T_lhs.shape == T_delta_ts.shape

        T_V_cts.append(np.max(T_lhs))
        # print("{}: Integral={}".format(ii, T_integral))
    T_V_cts = np.stack(T_V_cts, axis=0)

    ####################################################################
    logger.info("Plotting...")
    plot_dir = get_script_plot_dir()

    fig, axes = plt.subplots(2, layout="constrained")
    axes[0].plot(T_t, terms.Th_disc_int_lhs, color="C0", label="Discr")
    axes[0].plot(T_t, Th_disc_int, color="C5", ls="-.", lw=0.5, label="Discr2", zorder=10)
    axes[0].plot(T_t, Th_disc_int_trap, color="C4", label="Trap")
    axes[0].plot(T_t, T_int_cts, color="C1", ls="--", lw=2.5, alpha=0.8, label="Tsit5")
    # ax.plot(T_t, h(T_t), color="C3", ls="--", label="h")
    axes[0].legend()

    axes[1].plot(T_t, T_int_cts - terms.Th_disc_int_lhs, color="C0")
    axes[1].plot(T_t, T_int_cts - Th_disc_int_trap, color="C4")
    axes[1].set_yscale("log")

    fig.savefig(plot_dir / "test_compute_disc_avoid.pdf")

    ####################################################################
    fig, ax = plt.subplots(layout="constrained")
    ax.plot(T_t, T_V_cts, color="C1", lw=2.5, alpha=0.8, label="Tsit5")
    ax.plot(T_t, Th_V, color="C5", ls="-.", lw=0.5, label="Discr2", zorder=10)
    ax.plot(T_t, T_h, color="C3", ls="--", alpha=0.6, label="h")
    ax.legend()
    fig.savefig(plot_dir / "test_compute_V_disc_avoid.pdf")

    logger.info("Done!")


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
