import functools as ft

import ipdb
import jax.numpy as jnp
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import CenteredNorm

from pncbf.dyn.sim_cts_pbar import SimCtsPbar
from pncbf.dyn.unst_dbint import UnstDbInt
from pncbf.ncbf.compute_disc_avoid import compute_disc_avoid_h
from pncbf.nclf.sim_nclf import SimNCLF
from pncbf.utils.jax_utils import jax2np, jax_jit, jax_vmap, rep_vmap
from pncbf.utils.path_utils import mkdir
from pncbf.utils.paths import get_script_plot_dir
from pncbf.utils.schedules import lam_to_horizon


def main():
    plot_dir = get_script_plot_dir()
    task = UnstDbInt()

    n_steps = 150
    dt = task.dt

    # x0 = np.array([-0.6, 0.5])
    x0 = np.array([1.0, 0.0])
    pol = task.nom_pol_zero

    bb_x, bb_Xs, bb_Ys = task.get_contour_x0()

    ##############################################################
    h_labels = task.h_labels + ["max"]
    bbh_h = rep_vmap(task.h_components, rep=2)(bb_x)
    bb1_hmax = np.max(bbh_h, axis=2, keepdims=True)
    bbh_h = np.concatenate([bbh_h, bb1_hmax], axis=2)

    figsize = np.array([2 * task.nh, 2])
    fig, axes = plt.subplots(1, task.nh + 1, figsize=figsize, layout="constrained")
    for ii, ax in enumerate(axes):
        ax.contourf(bb_Xs, bb_Ys, bbh_h[:, :, ii], levels=11, norm=CenteredNorm(), cmap="RdBu_r")
        ax.set_title(h_labels[ii])
        task.plot_phase(ax)
    fig.savefig(plot_dir / "h_components.pdf")
    ##############################################################

    def get_xdot(state):
        return task.xdot(state, pol(state))

    bb_xdot = jax2np(jax_jit(rep_vmap(get_xdot, rep=2))(bb_x))

    def get_Th(state0):
        # sim = SimNCLF(task, pol, T)
        # T_x, _, T_t = sim.rollout_plot(state0)
        sim = SimCtsPbar(task, pol, n_steps, dt, max_steps=n_steps, use_pid=False, solver="bosh3", n_updates=2)
        T_x, T_t = sim.rollout_plot(state0)
        Th_h = jax_vmap(task.h_components)(T_x)
        return T_x, Th_h, T_x, T_t

    T_x, Th_h, T_x, T_t = get_Th(x0)

    _, bbTh_h, _, _ = jax2np(jax_jit(rep_vmap(get_Th, rep=2))(bb_x))

    bbT_h = np.max(bbTh_h, axis=3)
    # First index where
    bb_ttc_idx = np.argmax(bbT_h >= 0, axis=2)
    max_ttc = T_t[bb_ttc_idx.max()]
    print("Max ttc: {:.2f}s".format(max_ttc))

    def get_Vh_disc(lambd):
        bbVh_h = rep_vmap(ft.partial(compute_disc_avoid_h, lambd, dt), rep=2)(bbTh_h)
        return bbVh_h

    L_lambds = np.array([0.0, 0.25, 0.5, 1.0])
    Lbbh_Vh = jax2np(jax_jit(jax_vmap(get_Vh_disc))(L_lambds))

    Lbb1_Vhmax = np.max(Lbbh_Vh, axis=3, keepdims=True)
    Lbbh_Vh = np.concatenate([Lbbh_Vh, Lbb1_Vhmax], axis=3)

    n_lambds = len(L_lambds)

    vmin, vmax = Lbbh_Vh.min(), Lbbh_Vh.max()
    norm = CenteredNorm(halfrange=np.maximum(1.001 * vmax - 0, 0 - vmin * 1.001))

    ax: plt.Axes
    figsize = (2 * n_lambds, 2.0 * task.nh)
    fig, axes = plt.subplots(task.nh + 1, n_lambds, figsize=figsize, layout="constrained")
    for ii in range(n_lambds):
        streamplot_c = matplotlib.colors.to_rgba("C3", 0.2)
        for jj, ax in enumerate(axes[:, ii]):
            ax.plot(T_x[:, 0], T_x[:, 1], "C1", linewidth=1.0, zorder=10)
            ax.streamplot(
                bb_Xs,
                bb_Ys,
                bb_xdot[:, :, 0],
                bb_xdot[:, :, 1],
                color=streamplot_c,
                linewidth=0.5,
                zorder=5,
                integration_direction="forward",
            )
            task.plot_phase(ax)
            # axes[jj, ii].contourf(
            #     bb_Xs, bb_Ys, Lbbh_Vh[ii, :, :, jj], levels=11, norm=norm, cmap="RdBu_r", zorder=3.5, alpha=0.9
            # )
            axes[jj, ii].contour(bb_Xs, bb_Ys, Lbbh_Vh[ii, :, :, jj], levels=[0.0], colors=["C5"], zorder=10.0)

        lam_T = lam_to_horizon(L_lambds[ii], task.dt)
        axes[0, ii].set_title(rf"$\lambda={L_lambds[ii]}$" + "\nT={:.0f}".format(round(lam_T)))

    fig.savefig(plot_dir / "check_discr.pdf")


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
