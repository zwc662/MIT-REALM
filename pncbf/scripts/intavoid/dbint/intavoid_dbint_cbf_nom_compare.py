import functools as ft
import pathlib

import ipdb
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import typer
from loguru import logger
from matplotlib.colors import CenteredNorm

import run_config.int_avoid.doubleintwall_cfg
import run_config.int_avoid.segway_cfg
from pncbf.dyn.doubleint_wall import DoubleIntWall
from pncbf.dyn.sim_cts_pbar import SimCtsPbar
from pncbf.ncbf.int_avoid import IntAvoid
from pncbf.plotting.ez_line_collection import ez_line_collection
from pncbf.utils.ckpt_utils import get_id_from_ckpt, get_run_path_from_ckpt, load_ckpt_with_step
from pncbf.utils.jax_utils import jax2np, jax_default_x32, jax_jit, jax_vmap, rep_vmap
from pncbf.utils.logging import set_logger_format
from pncbf.utils.path_utils import mkdir


def main(ckpt_path: pathlib.Path):
    jax_default_x32()
    set_logger_format()
    task = DoubleIntWall()

    run_path = get_run_path_from_ckpt(ckpt_path)
    plot_dir = mkdir(run_path / "eval")

    nom_pol = task.nom_pol_osc

    CFG = run_config.int_avoid.doubleintwall_cfg.get(0)
    alg: IntAvoid = IntAvoid.create(0, task, CFG.alg_cfg, nom_pol)
    alg, ckpt_path = load_ckpt_with_step(alg, ckpt_path)
    logger.info("Loaded ckpt from {}!".format(ckpt_path))
    cid = get_id_from_ckpt(ckpt_path)

    n_steps = 160
    dt = task.dt

    #############################################################################
    b_x0 = task.get_plot_x0()
    bb_x, bb_Xs, bb_Ys = task.get_contour_x0()

    nom_pols = {
        "rng3": task.nom_pol_rng3,
        "rng3_shift": task.nom_pol_rng3,
        "zero": lambda x: np.zeros(task.nu),
        "zero_shift": lambda x: np.zeros(task.nu),
    }
    for pol_name, nom_pol in nom_pols.items():
        logger.info("Plotting nominal policy {}".format(pol_name))

        V_shift = 0
        max_ttc = 7.20
        if "shift" in pol_name:
            V_shift = -(1 - np.exp(-alg.lam * max_ttc)) * task.h_min

        alpha_safe, alpha_unsafe = 2.0, 100.0
        pol = ft.partial(alg.get_cbf_control_sloped, alpha_safe, alpha_unsafe, nom_pol=nom_pol, V_shift=V_shift + 1e-3)

        def get_xdot(state):
            return task.xdot(state, pol(state))

        sim = SimCtsPbar(
            task, pol, n_steps, dt, dt0=dt, use_obs=False, use_pid=False, max_steps=n_steps, solver="bosh3"
        )
        bT_x, _ = jax2np(jax_jit(jax_vmap(sim.rollout_plot))(b_x0))
        bb_xdot = jax2np(jax_jit(rep_vmap(get_xdot, rep=2))(bb_x))

        bb_Vh_pred = jax2np(jax_jit(rep_vmap(alg.get_V, rep=2))(bb_x))
        bb_Vh_pred = bb_Vh_pred + V_shift

        bbT_x, _ = jax2np(jax_jit(rep_vmap(sim.rollout_plot, rep=2))(bb_x))
        bbT_h = jax2np(rep_vmap(task.h, rep=3)(bbT_x))
        bb_Vh = bbT_h.max(axis=2)

        fig, ax = plt.subplots()
        task.plot_phase(ax)
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        ez_line_collection(bT_x, ax=ax, colors="C1", lw=0.5)
        ax.scatter(bT_x[:, -1, 0], bT_x[:, -1, 1], color="C5", s=1.0, zorder=10)

        streamplot_c = matplotlib.colors.to_rgba("C3", 0.7)
        ax.streamplot(bb_Xs, bb_Ys, bb_xdot[:, :, 0], bb_xdot[:, :, 1], color=streamplot_c, linewidth=0.5, zorder=5)

        cs0 = ax.contourf(bb_Xs, bb_Ys, bb_Vh, levels=11, norm=CenteredNorm(), cmap="RdBu_r", zorder=3.5, alpha=0.5)
        cbar = fig.colorbar(cs0, ax=ax, shrink=0.8)
        cs1 = ax.contour(bb_Xs, bb_Ys, bb_Vh, levels=[0.0], colors=["C5"], alpha=0.98, linewidths=1.5, zorder=4)
        cbar.add_lines(cs1)
        cs2 = ax.contour(bb_Xs, bb_Ys, bb_Vh_pred, levels=[0.0], colors=["C4"], alpha=0.98, linewidths=1.0, zorder=5)
        ax.set(xlim=xlim, ylim=ylim)
        fig.savefig(plot_dir / "nompol_{}.pdf".format(pol_name))
        plt.close(fig)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)
