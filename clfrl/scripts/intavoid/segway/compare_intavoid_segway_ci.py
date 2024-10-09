import functools as ft
import pathlib

import ipdb
import jax
import matplotlib.pyplot as plt
import numpy as np
import typer
from loguru import logger

import run_config.int_avoid.segway_cfg
import run_config.nclf.pend_cfg
import run_config.nclf_pol.pend_cfg
from clfrl.dyn.segway import Segway
from clfrl.dyn.sim_cts import SimCtsReal
from clfrl.ncbf.int_avoid import IntAvoid
from clfrl.plotting.contour_utils import centered_norm
from clfrl.plotting.legend_helpers import lline
from clfrl.plotting.plotstyle import PlotStyle
from clfrl.utils.ckpt_utils import get_id_from_ckpt, get_run_path_from_ckpt, load_ckpt
from clfrl.utils.jax_utils import jax2np, jax_jit, rep_vmap
from clfrl.utils.logging import set_logger_format
from clfrl.utils.path_utils import mkdir


def main(ckpt_path: pathlib.Path):
    set_logger_format()
    seed = 0

    run_path = get_run_path_from_ckpt(ckpt_path)
    plot_dir = mkdir(run_path / "compare_ci")

    task = Segway()
    CFG = run_config.int_avoid.segway_cfg.get(seed)
    nom_pol = task.nom_pol_lqr

    alg: IntAvoid = IntAvoid.create(seed, task, CFG.alg_cfg, nom_pol)
    alg = load_ckpt(alg, ckpt_path)
    logger.info("Loaded ckpt from {}!".format(ckpt_path))
    cid = get_id_from_ckpt(ckpt_path)

    pols = {
        # "QP(2)": ft.partial(alg.get_cbf_control_sloped, 2.0, 100.0),
        "QP(5)": ft.partial(alg.get_cbf_control_sloped, 5.0, 100.0),
        "QP(10)": ft.partial(alg.get_cbf_control_sloped, 10.0, 100.0),
        "Nom": nom_pol,
    }
    colors = {"QP(2)": "C6", "QP(5)": "C4", "QP(10)": "C1", "Nom": "C5"}

    tf = 8.0
    dt = 0.05
    n_pts = 80
    n_steps = int(round(tf / dt)) + 1
    logger.info("n_steps min: {}".format(tf / dt))
    logger.info("tf: {}, dt: {}".format(task.dt * alg.cfg.eval_cfg.eval_rollout_T, task.dt))

    levels = [0]
    linestyles = ["-"]

    def get_bb_Vh(pol, setup_idx: int):
        bb_x, bb_Xs, bb_Ys = task.get_contour_x0(setup_idx, n_pts=n_pts)
        sim = SimCtsReal(task, pol, tf, dt, dt0=dt, use_obs=False, use_pid=False, max_steps=n_steps)
        bbT_x, _, _ = rep_vmap(sim.rollout_plot, rep=2)(bb_x)
        bbT_h = rep_vmap(task.h, rep=3)(bbT_x)
        return bbT_h.max(axis=-1)

    bb_Vh_dict = {}
    for ii, setup in enumerate(task.phase2d_setups()):
        logger.info(f"Plotting {setup.plot_name}...")

        for pol_name, pol in pols.items():
            logger.info(f"    {pol_name}...")
            bb_h = jax2np(jax_jit(ft.partial(get_bb_Vh, pol, ii))())
            bb_Vh_dict[pol_name] = bb_h

        bb_x, bb_Xs, bb_Ys = task.get_contour_x0(n_pts=n_pts)
        fig, ax = plt.subplots(layout="constrained")
        for pol_name, bb_Vh in bb_Vh_dict.items():
            color = colors[pol_name]
            ax.contour(bb_Xs, bb_Ys, bb_Vh, levels=levels, linestyles=linestyles, colors=[color], zorder=3.5)
        setup.plot(ax)

        lines = [lline(color) for color in colors.values()]
        labels = list(colors.keys())
        ax.legend(lines, labels, loc="upper right")
        fig.savefig(plot_dir / f"compare_ci_{setup.plot_name}{cid}.pdf")
        plt.close(fig)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)
