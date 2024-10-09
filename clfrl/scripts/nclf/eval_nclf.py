import functools as ft
import pathlib

import ipdb
import matplotlib.pyplot as plt
import numpy as np
import typer
from loguru import logger
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

import run_config.nclf.pend_cfg
import wandb
from clfrl.dyn.pend import Pend
from clfrl.nclf.nclf import NCLF
from clfrl.plotting.plot_task_summary import plot_task_summary
from clfrl.plotting.plotter import Plotter
from clfrl.training.ckpt_manager import get_ckpt_manager, save_create_args
from clfrl.training.run_dir import init_wandb_and_get_run_dir
from clfrl.utils.ckpt_utils import get_run_path_from_ckpt, load_ckpt
from clfrl.utils.jax_utils import jax2np, jax_jit, jax_vmap, rep_vmap
from clfrl.utils.logging import set_logger_format


def main(ckpt_path: pathlib.Path):
    set_logger_format()
    seed = 0

    task = Pend()
    CFG = run_config.nclf.pend_cfg.get(seed)
    CFG.alg_cfg.eval_cfg.eval_rollout_T = 128
    task._dt = 0.02
    alg: NCLF = NCLF.create(seed, task, CFG.alg_cfg)
    alg = load_ckpt(alg, ckpt_path)
    logger.info("Loading ckpt from {}... Done!".format(ckpt_path))

    eval_data: NCLF.EvalData = jax2np(alg.eval(interp=0.2))

    # b_V_plot0 = jax2np(jax_vmap(ft.partial(alg.get_V, alg.V.params))(eval_data.bT_x_plot[:, 0]))
    # # Ignore all pts with V larger than thresh.
    pos = eval_data.bT_x_plot
    is_bad = (np.abs(pos[:, 0, 1]) > 5) | (np.abs(pos[:, 0, 0] - np.pi) < 1)
    pos = pos[~is_bad]

    ############################
    run_path = get_run_path_from_ckpt(ckpt_path)
    plot_dir = run_path / "plots"
    fig, ax = plt.subplots(layout="constrained", dpi=300)
    PLOT_XMIN, PLOT_XMAX = -2 * np.pi - 2.5, 4 * np.pi + 2.5
    PLOT_YMIN, PLOT_YMAX = -20.0, 20.0
    # ax.set(xlim=(PLOT_XMIN, PLOT_XMAX), ylim=(PLOT_YMIN, PLOT_YMAX))
    ax.set(xlim=(PLOT_XMIN, PLOT_XMAX))

    # Horizontal and vertical line at the goal.
    [ax.axvline(np.pi + 2 * ii * np.pi, zorder=2.5) for ii in [-1, 0, 1]]
    ax.axhline(0.0, zorder=2.5)

    # Plot trajs.
    valid_color = "C1"
    end_style = dict(s=1**2, zorder=7, marker="o")

    valid_col = LineCollection(pos, lw=0.3, zorder=5, colors=valid_color)
    ax.add_collection(valid_col)
    ax.scatter(pos[:, -1, 0], pos[:, -1, 1], color="C5", **end_style)
    ax.scatter(pos[:, 0, 0], pos[:, 0, 1], color="black", s=1**2, zorder=6, marker="s")
    ax.autoscale_view()

    # Plot contour beneath.
    xlim, ylim = np.array(ax.get_xlim()), ax.get_ylim()
    xlim = xlim.clip(0, 2 * np.pi)
    b_xs, b_ys = np.linspace(*xlim, num=96), np.linspace(*ylim, num=96)
    bb_Xs, bb_Ys = np.meshgrid(b_xs, b_ys)
    bb_states = np.stack([bb_Xs, bb_Ys], axis=-1)
    bb_V = jax2np(jax_jit(rep_vmap(ft.partial(alg.get_V, alg.V.params), rep=2))(bb_states))
    norm = Normalize(vmin=0.0, vmax=bb_V.max())
    levels = 15
    cs0 = ax.contourf(bb_Xs, bb_Ys, bb_V, levels=levels, norm=norm, cmap="rocket", alpha=0.5)
    cs0 = ax.contourf(bb_Xs - 4 * np.pi, bb_Ys, bb_V, levels=levels, norm=norm, cmap="rocket", alpha=0.5)
    cs0 = ax.contourf(bb_Xs - 2 * np.pi, bb_Ys, bb_V, levels=levels, norm=norm, cmap="rocket", alpha=0.5)
    cs0 = ax.contourf(bb_Xs + 2 * np.pi, bb_Ys, bb_V, levels=levels, norm=norm, cmap="rocket", alpha=0.5)
    cs0 = ax.contourf(bb_Xs + 4 * np.pi, bb_Ys, bb_V, levels=levels, norm=norm, cmap="rocket", alpha=0.5)
    fig.colorbar(cs0, ax=ax, shrink=0.8)

    # Plot minimum.
    min_idx = np.unravel_index(bb_V.argmin(), bb_V.shape)
    ax.plot([bb_Xs[min_idx]], [bb_Ys[min_idx]], marker="*", color="C4", zorder=3.0)

    fig.savefig(plot_dir / "eval_phase.pdf")
    fig.savefig(plot_dir / "eval_phase.png")

    # Plot a closeup around the goal.
    fig, ax = plt.subplots(layout="constrained", dpi=300)
    dx, dy = 0.2, 0.5
    goal_pt = task.goal_pt()
    b_xs = np.linspace(goal_pt[0] - dx, goal_pt[0] + dx, num=129)
    b_ys = np.linspace(goal_pt[1] - dy, goal_pt[1] + dy, num=129)
    bb_Xs, bb_Ys = np.meshgrid(b_xs, b_ys)
    bb_states = np.stack([bb_Xs, bb_Ys], axis=-1)
    bb_V = jax2np(jax_jit(rep_vmap(ft.partial(alg.get_V, alg.V.params), rep=2))(bb_states))
    norm = Normalize(vmin=0.0, vmax=bb_V.max())
    levels = 35
    cs0 = ax.contourf(bb_Xs, bb_Ys, bb_V, levels=levels, norm=norm, cmap="rocket")
    fig.colorbar(cs0, ax=ax, shrink=0.8)
    # Plot minimum.
    min_idx = np.unravel_index(bb_V.argmin(), bb_V.shape)
    ax.plot([bb_Xs[min_idx]], [bb_Ys[min_idx]], marker="*", color="C4", zorder=3.0)

    fig.savefig(plot_dir / "V_zoom.pdf")
    fig.savefig(plot_dir / "V_zoom.png")


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)
