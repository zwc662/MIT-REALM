import functools as ft

import ipdb
import jax.numpy as jnp
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import CenteredNorm

from clfrl.dyn.doubleint_wall import DoubleIntWall
from clfrl.ncbf.compute_disc_avoid import compute_disc_avoid_h
from clfrl.nclf.sim_nclf import SimNCLF
from clfrl.utils.jax_utils import jax2np, jax_jit, jax_vmap, rep_vmap
from clfrl.utils.path_utils import mkdir
from clfrl.utils.paths import get_script_plot_dir


def main():
    plot_dir = mkdir(get_script_plot_dir() / "doubleint")
    task = DoubleIntWall()

    T = 160
    x0 = np.array([-0.6, 1.5])

    def pol(state):
        p, v = task.chk_x(state)
        tmp = jnp.sign(p) * jnp.minimum(jnp.sign(p) * v, 0.0) ** 2 / 2.0 <= p
        return jnp.array([jnp.where(tmp, -1.0, 1.0)])

    bb_x, bb_Xs, bb_Ys = task.get_contour_x0()

    def get_xdot(state):
        return task.xdot(state, pol(state))

    bb_xdot = jax2np(jax_jit(rep_vmap(get_xdot, rep=2))(bb_x))

    def get_Th(state0):
        sim = SimNCLF(task, pol, T)
        T_x, _, T_t = sim.rollout_plot(state0)
        Th_h = jax_vmap(task.h_components)(T_x)
        return Th_h, T_x, T_t

    Th_h, T_x, T_t = get_Th(x0)
    dt = np.diff(T_t).mean()

    bbTh_h, _, _ = jax_jit(rep_vmap(get_Th, rep=2))(bb_x)

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
    figsize = (2 * n_lambds, 6)
    fig, axes = plt.subplots(3, n_lambds, figsize=figsize, layout="constrained")
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
            axes[jj, ii].contourf(
                bb_Xs, bb_Ys, Lbbh_Vh[ii, :, :, jj], levels=11, norm=norm, cmap="RdBu_r", zorder=3.5, alpha=0.9
            )
            axes[jj, ii].contour(bb_Xs, bb_Ys, Lbbh_Vh[ii, :, :, jj], levels=[0.0], colors=["C5"], zorder=10.0)

        axes[0, ii].set_title(rf"$\lambda = {L_lambds[ii]}$")

    fig.savefig(plot_dir / "check_discr.pdf")


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
