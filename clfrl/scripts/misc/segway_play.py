import functools as ft

import ipdb
import matplotlib.pyplot as plt
import numpy as np

from clfrl.dyn.segway import Segway
from clfrl.dyn.sim_cts import SimCts, integrate
from clfrl.utils.jax_utils import jax2np, jax_jit, jax_use_cpu
from clfrl.utils.path_utils import mkdir
from clfrl.utils.paths import get_script_plot_dir


def main():
    plot_dir = mkdir(get_script_plot_dir() / "segway")
    jax_use_cpu()

    task = Segway()

    T = 800
    dt = task.dt

    # x0 = np.array([0.0, -0.9, 0.0, 0.0])
    # x0 = np.array([0.0, -0.79, 0.0, 0.0])
    # x0 = np.array([0.0, -0.5, 0.0, 0.0])
    x0 = np.array([0.0, 1.05, 0.0, -1.0])

    # u_traj = np.zeros((T, task.nu))
    # u_traj[:20, :] = 1.0
    # u_traj[20:40, :] = -10.0
    # u_traj[10:, :] = 1.0
    # u_traj[2, :] = 1.0
    # u_traj[3, :] = 0.2
    # u_traj[4:, :] = 0.0155
    # int_fn = ft.partial(integrate, 4, dt, task.xdot)
    # T_x, T_t = int_fn(x0, u_traj)

    interp_pts = 4
    sim = SimCts(task, task.nom_pol_lqr, T, interp_pts, pol_dt=dt, use_obs=False)
    T_x, T_u, T_t = jax2np(jax_jit(sim.rollout_plot)(x0))

    T_t_u = np.arange(T) * dt

    x_labels = task.x_labels

    fig, axes = plt.subplots(task.nx + 1, layout="constrained")
    # axes[1].axhline(np.pi, color="C2")
    for ii, ax in enumerate(axes[:-1]):
        ax.plot(T_t, T_x[:, ii], marker="o", ms=1.5, color="C1")
        ax.set_title(x_labels[ii])
    for t_u in T_t_u[:5]:
        axes[1].axvline(t_u, color="C3", alpha=0.3)
    axes[-1].plot(T_t_u, T_u[:, 0], marker="o", ms=1.5, color="C1")
    # axes[-1].plot(T_t_u, u_traj[:, 0], marker="o", ms=1.5, color="C1")
    axes[-1].set_title(r"$F$")
    fig.savefig(plot_dir / "play.pdf")

    # Save the traj so we can animate it.
    np.savez(plot_dir / "play.npz", T_x=T_x, T_t=T_t)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
