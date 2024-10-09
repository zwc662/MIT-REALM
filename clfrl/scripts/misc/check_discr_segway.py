import functools as ft

import ipdb
import jax.numpy as jnp
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.colors import CenteredNorm, Normalize

from clfrl.dyn.segway import Segway
from clfrl.ncbf.compute_disc_avoid import compute_disc_avoid_h
from clfrl.nclf.sim_nclf import SimNCLF
from clfrl.utils.jax_utils import jax2np, jax_jit, jax_vmap, merge01, rep_vmap
from clfrl.utils.path_utils import mkdir
from clfrl.utils.paths import get_script_plot_dir


def main():
    plot_dir = mkdir(get_script_plot_dir() / "segway")
    task = Segway()

    T = 800
    x0 = np.array([0.0, -0.5, 0.0, 0.0])

    pol = task.nom_pol_lqr

    # L_lambds = np.array([0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0])
    # L_lambds = np.array([0.0, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0])
    # L_lambds = np.array([0.0, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5])
    L_lambds = np.array([0.0, 0.05, 0.1, 0.25, 0.5, 1.0])
    L_gammas = np.exp(-L_lambds * task.dt)
    L_H = 1 / (1 - L_gammas)
    n_lambds = len(L_lambds)

    def get_xdot(state):
        return task.xdot(state, pol(state))

    for ii, setup in enumerate(task.phase2d_setups()):
        logger.info("Plotting {}...".format(setup.plot_name))

        bb_x, bb_Xs, bb_Ys = task.get_contour_x0(setup=ii, n_pts=256)
        # bb_xdot = jax2np(jax_jit(rep_vmap(get_xdot, rep=2))(bb_x))

        def get_Th(state0):
            sim = SimNCLF(task, pol, T)
            T_x, _, T_t = sim.rollout_plot(state0)
            Th_h = jax_vmap(task.h_components)(T_x)
            return Th_h, T_x, T_t

        Th_h, T_x, T_t = get_Th(x0)
        dt = np.diff(T_t).mean()
        n_real_steps_per_sim_step = dt / task.dt

        bbTh_h, bbT_x, _ = jax2np(jax_jit(rep_vmap(get_Th, rep=2))(bb_x))

        # See what the bounds on velocity are inside the control invariant region.
        bTh_h, bT_x = merge01(bbTh_h), merge01(bbT_x)
        b_is_safe = np.all(bTh_h.max(axis=2) <= 0.0, axis=1)
        bT_x_safe = bT_x[b_is_safe]

        xmin, xmax = bT_x_safe.min(axis=(0, 1)), bT_x_safe.max(axis=(0, 1))
        logger.info("x_min: {}".format(xmin))
        logger.info("x_max: {}".format(xmax))

        # Plot how many timesteps before constraint violation.
        def get_t_violate(T_h):
            assert T_h.ndim == 1
            has_violate = jnp.any(T_h > 0.0)
            t_violate = jnp.argmax(T_h >= 0.0)
            return jnp.where(has_violate, t_violate, jnp.inf) * n_real_steps_per_sim_step

        bbh_t_viol = jax2np(jax_jit(rep_vmap(jax_vmap(get_t_violate, in_axes=1), rep=2))(bbTh_h))
        bb1_t_viol_min = np.min(bbh_t_viol, axis=2, keepdims=True)
        bbh_t_viol = np.concatenate([bbh_t_viol, bb1_t_viol_min], axis=2)
        bbh_t_viol = np.ma.array(bbh_t_viol, mask=~np.isfinite(bbh_t_viol))

        t_viol_min = bb1_t_viol_min.flatten()
        t_viol_min = t_viol_min[np.isfinite(t_viol_min)]

        fig, ax = plt.subplots(layout="constrained")
        ax.hist(t_viol_min, bins=48)
        fig.savefig(plot_dir / f"time_to_violate_dist_{setup.plot_name}.pdf")
        plt.close(fig)

        norm = Normalize(vmin=0, vmax=1.005 * t_viol_min.max())
        figsize = (2 * (task.nh + 1), 2)
        fig, axes = plt.subplots(1, task.nh + 1, figsize=figsize, layout="constrained")
        for ii, ax in enumerate(axes):
            cs0 = ax.contourf(bb_Xs, bb_Ys, bbh_t_viol[:, :, ii], norm=norm, levels=15, cmap="rocket_r")
            setup.plot(ax)
            if ii < task.nh:
                ax.set_title(task.h_labels[ii])
            else:
                ax.set_title("max: {:.1f}".format(bbh_t_viol[:, :, ii].max()))

        fig.colorbar(cs0, ax=axes.ravel().tolist(), shrink=0.95)
        fig.savefig(plot_dir / f"time_to_violate_{setup.plot_name}.pdf")
        plt.close(fig)

        def get_Vh_disc(lambd):
            bbVh_h = rep_vmap(ft.partial(compute_disc_avoid_h, lambd, dt), rep=2)(bbTh_h)
            return bbVh_h

        Lbbh_Vh = jax2np(jax_jit(jax_vmap(get_Vh_disc))(L_lambds))

        Lbb1_Vhmax = np.max(Lbbh_Vh, axis=3, keepdims=True)
        Lbbh_Vh = np.concatenate([Lbbh_Vh, Lbb1_Vhmax], axis=3)

        vmin, vmax = Lbbh_Vh.min(), Lbbh_Vh.max()
        norm = CenteredNorm(halfrange=np.maximum(1.001 * vmax - 0, 0 - vmin * 1.001))

        ax: plt.Axes
        figsize = (2 * n_lambds, 2 * 5)
        fig, axes = plt.subplots(5, n_lambds, figsize=figsize, layout="constrained")
        for ii in range(n_lambds):
            for jj, ax in enumerate(axes[:, ii]):
                ax.plot(T_x[:, setup.idx0], T_x[:, setup.idx1], "C1", linewidth=1.0, zorder=10)
                setup.plot(ax)
                axes[jj, ii].contourf(
                    bb_Xs, bb_Ys, Lbbh_Vh[ii, :, :, jj], levels=11, norm=norm, cmap="RdBu_r", zorder=3.5, alpha=0.9
                )
                axes[jj, ii].contour(
                    bb_Xs, bb_Ys, Lbbh_Vh[ii, :, :, jj], levels=[0.0], linewidths=[1.0], colors=["C5"], zorder=10.0
                )

                if ii > 0:
                    # Plot the true one.
                    axes[jj, ii].contour(
                        bb_Xs,
                        bb_Ys,
                        Lbbh_Vh[0, :, :, jj],
                        levels=[0.0],
                        colors=["C2"],
                        linestyles=["--"],
                        alpha=0.7,
                        zorder=8.0,
                    )

            title = "\n".join([rf"$\lambda={L_lambds[ii]}$", rf"$H={L_H[ii]:.0f}$"])
            axes[0, ii].set_title(title)

        fig.savefig(plot_dir / f"check_discr_{setup.plot_name}.pdf")


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
