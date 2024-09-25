import einops as ei
import ipdb
import matplotlib.pyplot as plt
import numpy as np

from pncbf.dyn.f16_two import F16Two
from pncbf.dyn.sim_cts_pbar import SimCtsPbar
from pncbf.utils.jax_utils import jax_default_x32, jax_jit_np, jax_vmap
from pncbf.utils.logging import set_logger_format
from pncbf.utils.paths import get_script_plot_dir


def main():
    plot_dir = get_script_plot_dir()

    jax_default_x32()
    set_logger_format()
    task = F16Two()

    tf = 20.0
    dt = task.dt / 2
    n_steps = int(round(tf / dt))

    b = 16

    x0 = task.nominal_val_state()
    b_x0 = ei.repeat(x0, "nx -> b nx", b=b)

    b_x0[:, task.PE0] = np.linspace(-2000, -1000, b)
    b_x0[:, task.PN0] = np.linspace(-250, 250, b)

    sim = SimCtsPbar(
        task, task.nom_pol_N0_pid, n_steps, dt, dt0=dt, use_obs=False, use_pid=False, max_steps=n_steps, solver="bosh3"
    )
    bT_x, bT_t = jax_jit_np(jax_vmap(sim.rollout_plot))(b_x0)

    # Plot on pos2d.
    fig, ax = plt.subplots(layout="constrained")
    for ii in range(b):
        ax.plot(bT_x[ii, :, task.PE0], bT_x[ii, :, task.PN0], color="C1", lw=0.5, alpha=0.9, zorder=8)
        ax.plot(bT_x[ii, 0, task.PE0], bT_x[ii, 0, task.PN0], marker="s", ms=5, color="C1")
    # task.plot_pos2d(ax)
    ax.set(xlabel="East", ylabel="North", aspect="equal")
    fig.savefig(plot_dir / f"pos2d_nompid.pdf", bbox_inches="tight")
    plt.close(fig)
    #######################################################################################
    nrows = task.nx
    figsize = np.array([6, 1.0 * nrows])
    fig, axes = plt.subplots(nrows, figsize=figsize, sharex=True, layout="constrained")
    for ii, ax in enumerate(axes[: task.nx]):
        for jj in range(b):
            ax.plot(bT_t[0], bT_x[jj, :, ii], lw=1.0)
        ax.set_ylabel(task.x_labels[ii], rotation=0, ha="right")
    fig.savefig(plot_dir / f"pos2d_traj.pdf")
    plt.close(fig)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
