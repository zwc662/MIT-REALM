import functools as ft
import ipdb
import jax.numpy as jnp
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

from pncbf.dyn.odeint import tsit5_dense_pid
from pncbf.dyn.quadcircle import QuadCircle
from pncbf.dyn.sim_cts import SimCtsReal
from pncbf.ncbf.compute_disc_avoid import compute_all_disc_avoid_terms
from pncbf.utils.jax_utils import jax2np, jax_use_cpu, jax_vmap, rep_vmap
from pncbf.utils.paths import get_script_plot_dir


def main():
    plot_dir = get_script_plot_dir()
    jax_use_cpu()
    task = QuadCircle()

    tf = 3.0
    result_dt = 0.05

    # Integrate, see what h looks like under different discounts.
    x0 = task.nominal_val_state()
    sim = SimCtsReal(task, task.nom_pol_vf, tf, result_dt)
    T_states, T_t, _ = sim.rollout_plot(x0)
    Th_h = jax2np(jax_vmap(task.h_components)(T_states))

    # Compute discounted.
    b_lams = np.array([0.0, 0.1, 0.5, 1.0])
    bTh_h = jax_vmap(ft.partial(compute_all_disc_avoid_terms, dt=result_dt, Th_h=Th_h))(b_lams).Th_max_lhs

    # Plot.
    h_labels = task.h_labels

    figsize = np.array([5, task.nh * 2])
    fig, axes = plt.subplots(task.nh, figsize=figsize, sharex=True, layout="constrained")
    for ii, ax in enumerate(axes):
        ax.plot(T_t, Th_h[:, ii], color="C0", label=r"$h$")
        for jj, lam in enumerate(b_lams):
            ax.plot(T_t, bTh_h[jj, :, ii], color=f"C{jj+1}", label=f"{lam}")
        ax.set_title(h_labels[ii])
    axes[0].legend()
    fig.savefig(plot_dir / "check_disc.pdf")


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
