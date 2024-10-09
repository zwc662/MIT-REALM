import functools as ft
import pathlib

import ipdb
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import typer
from loguru import logger
from matplotlib.colors import CenteredNorm

import run_config.avoid_fixed.doubleintwall_cfg
import run_config.nclf.pend_cfg
import run_config.nclf_pol.pend_cfg
from clfrl.dyn.doubleint_wall import DoubleIntWall
from clfrl.dyn.segway import Segway
from clfrl.dyn.sim_cts import SimCts, integrate
from clfrl.ncbf.avoid_fixed import AvoidFixed
from clfrl.nclf.sim_nclf import SimNCLF
from clfrl.utils.ckpt_utils import load_ckpt
from clfrl.utils.jax_utils import jax2np, jax_jit, jax_vmap, rep_vmap
from clfrl.utils.path_utils import mkdir
from clfrl.utils.paths import get_script_plot_dir


def main():
    plot_dir = mkdir(get_script_plot_dir() / "segway")
    task = Segway()

    pols = {"lqr": task.nom_pol_lqr}

    dt = task.dt
    interp_pts = 2

    # x0 = np.array([0.0, -0.5, 0.0, 0.0])
    x0 = np.array([0.0, 0.6, 0.0, -1.0])

    setup_idx = 0

    Ts = [100, 200, 400, 800]

    for pol_name, pol in pols.items():
        bb_Vs = []
        for T in Ts:
            logger.info("Plotting T={}...".format(T))
            sim = SimCts(task, pol, T, interp_pts, pol_dt=dt)
            # One point.
            T_x_den, _, T_t_den = jax2np(jax_jit(sim.rollout_plot)(x0))

            # Also plot many points in grey to show the overall vector field maybe.
            # Also, integrate to see what the true value function looks like.
            bb_x, bb_Xs, bb_Ys = task.get_contour_x0(setup_idx)

            def get_xdot(state):
                return task.xdot(state, pol(state))

            bb_xdot = jax2np(jax_jit(rep_vmap(get_xdot, rep=2))(bb_x))

            def get_hmax(x0):
                sim = SimNCLF(task, pol, T)
                T_x, _, T_t = sim.rollout_plot(x0)
                T_h = jax_vmap(task.h)(T_x)
                return T_h.max()

            bb_V = jax_jit(rep_vmap(get_hmax, rep=2))(bb_x)

            ax: plt.Axes
            # fig, axes = plt.subplots(1, 2, layout="constrained")
            # ax = axes[0]
            fig, ax = plt.subplots(layout="constrained")
            ax.plot(T_x_den[:, task.TH], T_x_den[:, task.W], color="C1", zorder=5)

            bb_xdot_th, bb_xdot_omega = bb_xdot[:, :, task.TH], bb_xdot[:, :, task.W]

            streamplot_c = matplotlib.colors.to_rgba("C3", 0.7)
            ax.streamplot(bb_Xs, bb_Ys, bb_xdot_th, bb_xdot_omega, color=streamplot_c, linewidth=0.5, zorder=5)
            ax.set_title(pol_name)
            xlim, ylim = ax.get_xlim(), ax.get_ylim()
            task.plot(ax, setup_idx)
            ax.set(xlim=xlim, ylim=ylim)
            cs0 = ax.contourf(bb_Xs, bb_Ys, bb_V, levels=11, norm=CenteredNorm(), cmap="RdBu_r", zorder=3.5, alpha=0.5)
            cbar = fig.colorbar(cs0, ax=ax, shrink=0.8)
            if bb_V.min() < 0 and bb_V.max() > 0:
                cs1 = ax.contour(bb_Xs, bb_Ys, bb_V, levels=[0.0], colors=["C5"], alpha=0.98, linewidths=1.5, zorder=4)
                cbar.add_lines(cs1)
            fig.savefig(plot_dir / "segway_pol_{}_{}.pdf".format(pol_name, T))
            plt.close(fig)

            bb_Vs.append(bb_V)

        # Compare the level sets.
        fig, ax = plt.subplots(layout="constrained")
        for ii, (T, bb_V) in enumerate(zip(Ts, bb_Vs)):
            ax.contour(bb_Xs, bb_Ys, bb_V, levels=[0.0], colors=[f"C{ii}"], alpha=0.98, linewidths=1.5, zorder=4)
        lines = [plt.Line2D([0], [0], color=f"C{ii}", label=f"T={T}") for ii, T in enumerate(Ts)]
        ax.legend(lines, [f"T={T}" for T in Ts])
        fig.savefig(plot_dir / "segway_pol_{}_compare.pdf".format(pol_name))
        plt.close(fig)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
