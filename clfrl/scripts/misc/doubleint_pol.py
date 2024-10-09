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
from clfrl.dyn.sim_cts import SimCts
from clfrl.ncbf.avoid_fixed import AvoidFixed
from clfrl.nclf.sim_nclf import SimNCLF
from clfrl.utils.ckpt_utils import load_ckpt
from clfrl.utils.jax_utils import jax2np, jax_jit, jax_vmap, rep_vmap
from clfrl.utils.paths import get_script_plot_dir


def main(ckpt_path: pathlib.Path = typer.Argument(None)):
    task = DoubleIntWall()

    # pols = {"pp": task.nom_pol_osc, "random": task.nom_pol_rng, "random2": task.nom_pol_rng2}
    pols = {}

    if ckpt_path is not None:
        CFG = run_config.avoid_fixed.doubleintwall_cfg.get(0)
        alg: AvoidFixed = AvoidFixed.create(0, task, CFG.alg_cfg, task.nom_pol_osc)
        alg = load_ckpt(alg, ckpt_path)
        logger.info("Loaded ckpt from {}!".format(ckpt_path))

        pols["opt"] = alg.get_opt_u

    T = 160
    dt = 0.1
    interp_pts = 2

    x0 = np.array([-0.8, 1.5])

    for pol_name, pol in pols.items():
        sim = SimCts(task, pol, T, interp_pts, pol_dt=dt)
        # One point.
        T_x_den, _, T_t_den = jax2np(jax_jit(sim.rollout_plot)(x0))

        # Also plot many points in grey to show the overall vector field maybe.
        # Also, integrate to see what the true value function looks like.
        bb_x, bb_Xs, bb_Ys = task.get_contour_x0()

        def get_xdot(state):
            return task.xdot(state, pol(state))

        bb_xdot = jax2np(jax_jit(rep_vmap(get_xdot, rep=2))(bb_x))

        def get_hmax(x0):
            sim = SimNCLF(task, pol, T)
            T_x, _, _ = sim.rollout_plot(x0)
            T_h = jax_vmap(task.h)(T_x)
            return T_h.max()

        bb_V = jax_jit(rep_vmap(get_hmax, rep=2))(bb_x)

        plot_dir = get_script_plot_dir()

        ax: plt.Axes
        fig, ax = plt.subplots()
        ax.plot(T_x_den[:, 0], T_x_den[:, 1], color="C1", zorder=5)

        streamplot_c = matplotlib.colors.to_rgba("C3", 0.7)
        ax.streamplot(bb_Xs, bb_Ys, bb_xdot[:, :, 0], bb_xdot[:, :, 1], color=streamplot_c, linewidth=0.5, zorder=5)
        ax.set_title(pol_name)
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        task.plot_phase(ax)
        ax.set(xlim=xlim, ylim=ylim)
        cs0 = ax.contourf(bb_Xs, bb_Ys, bb_V, levels=11, norm=CenteredNorm(), cmap="RdBu_r", zorder=3.5, alpha=0.5)
        cbar = fig.colorbar(cs0, ax=ax, shrink=0.8)
        if bb_V.min() < 0 and bb_V.max() > 0:
            cs1 = ax.contour(bb_Xs, bb_Ys, bb_V, levels=[0.0], colors=["C5"], alpha=0.98, linewidths=1.5, zorder=4)
            cbar.add_lines(cs1)
        fig.savefig(plot_dir / "doubleint_pol_{}.pdf".format(pol_name))


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)
