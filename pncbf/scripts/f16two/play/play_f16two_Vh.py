import functools as ft
import pickle

import ipdb
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import typer
from loguru import logger
from matplotlib.colors import Normalize, CenteredNorm

from pncbf.dyn.f16_two import F16Two
from pncbf.dyn.sim_cts_pbar import SimCtsPbar
from pncbf.ncbf.compute_disc_avoid import compute_all_disc_avoid_terms
from pncbf.plotting.contour_utils import centered_norm
from pncbf.utils.jax_utils import jax_jit_np, jax_vmap, rep_vmap
from pncbf.utils.path_utils import mkdir
from pncbf.utils.paths import get_script_plot_dir
from pncbf.utils.schedules import lam_to_horizon


def main():
    plot_dir = get_script_plot_dir()
    task = F16Two()

    tf = 10.0
    dt = task.dt
    n_steps = int(round(tf / dt))

    x0 = task.nominal_val_state()
    x0[task.PSI0] = -np.pi / 2
    x0[task.PSI1] = +np.pi / 2

    sim = SimCtsPbar(task, task.nom_pol_pid, n_steps, dt, max_steps=n_steps, use_pid=False, solver="bosh3")
    T_x, T_t = sim.rollout_plot(x0)
    Th_h = jax_vmap(task.h_components)(T_x)
    T_h = Th_h.max(-1)

    fig, ax = plt.subplots()
    ax.plot(T_x[:, task.PE0], T_x[:, task.PN0], color="C1", label="Self")
    ax.plot(T_x[:, task.PE1], T_x[:, task.PN1], color="C0", label="Other", linestyle="--", zorder=5)
    ax.legend()
    task.plot_pos2d(ax)
    fig.savefig(plot_dir / "f16two_Vh.pdf")

    #################################################################
    nrows = task.nx + 1
    figsize = np.array([6, 1.0 * nrows])
    x_labels, h_labels = task.x_labels, task.h_labels
    fig, axes = plt.subplots(nrows, figsize=figsize, sharex=True, layout="constrained")
    for ii, ax in enumerate(axes[: task.nx]):
        ax.plot(T_t, T_x[:, ii], color="C1", lw=1.0)
        ax.set_ylabel(x_labels[ii], rotation=0, ha="right")
    axes[-1].plot(T_t, T_h, color="C0", lw=1.0)

    # axes[0].legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, 1.0), fontsize="x-small")
    fig.savefig(plot_dir / "f16two_traj.pdf")
    plt.close(fig)
    #################################################################
    # Visualize the h components.
    bb_x, bb_Xs, bb_Ys = task.get_contour_x0(n_pts=100)
    bbh_h = rep_vmap(task.h_components, rep=2)(bb_x)

    fig, ax = plt.subplots(layout="constrained")
    ax.contourf(bb_Xs, bb_Ys, bbh_h[:, :, 10], levels=21, norm=CenteredNorm(), cmap="RdBu_r")
    task.plot_pos2d(ax)
    fig.savefig(plot_dir / "h_collide.pdf")


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
