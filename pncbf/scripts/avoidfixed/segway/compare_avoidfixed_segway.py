import functools as ft
import pathlib

import ipdb
import matplotlib.pyplot as plt
import numpy as np
import typer
from loguru import logger

import run_config.avoid_fixed.doubleintwall_cfg
import run_config.avoid_fixed.segway_cfg
import run_config.nclf.pend_cfg
import run_config.nclf_pol.pend_cfg
from pncbf.dyn.segway import Segway
from pncbf.ncbf.avoid_fixed import AvoidFixed
from pncbf.nclf.sim_nclf import SimNCLF
from pncbf.plotting.legend_helpers import lline
from pncbf.utils.ckpt_utils import get_id_from_ckpt, get_run_path_from_ckpt, load_ckpt
from pncbf.utils.jax_utils import jax2np, jax_jit, rep_vmap
from pncbf.utils.logging import set_logger_format


def main(ckpt_path: pathlib.Path):
    set_logger_format()
    seed = 0

    run_path = get_run_path_from_ckpt(ckpt_path)
    plot_dir = run_path / "plots"

    task = Segway()
    CFG = run_config.avoid_fixed.segway_cfg.get(seed)
    CFG.alg_cfg.eval_cfg.eval_rollout_T = 600
    nom_pol = task.nom_pol_lqr

    alg: AvoidFixed = AvoidFixed.create(seed, task, CFG.alg_cfg, nom_pol)
    alg = load_ckpt(alg, ckpt_path)
    logger.info("Loaded ckpt from {}!".format(ckpt_path))
    cid = get_id_from_ckpt(ckpt_path)

    T = 600
    # x0 = np.array([0.0, 0.75, 0.0, -1.0])
    # x0 = np.array([0.0, 0.0, 0.0, -2.5])
    x0 = np.array([0.0, -0.5, 0.0, 0.0])

    logger.info("Rollout cbf")
    # alpha_safe, alpha_unsafe = 0.1, 100.0
    alpha_safe, alpha_unsafe = 2.0, 50.0
    sim = SimNCLF(task, ft.partial(alg.get_cbf_control_sloped, alpha_safe, alpha_unsafe), T)
    T_x, T_u, T_t = jax2np(jax_jit(sim.rollout_plot)(x0))

    # Also visualize the original nominal policy.
    logger.info("Rollout nom")
    sim = SimNCLF(task, nom_pol, T)
    T_x_nom, _, T_t_nom = jax2np(jax_jit(sim.rollout_plot)(x0))

    # Also visualize the bang-bang policy.
    logger.info("Rollout opt_u")
    sim = SimNCLF(task, alg.get_opt_u, T)
    T_x_opt, _, T_t_nom = jax2np(jax_jit(sim.rollout_plot)(x0))

    for ii, setup in enumerate(task.phase2d_setups()):
        logger.info("Get nom")
        bb_x, bb_Xs, bb_Ys = task.get_contour_x0(ii)
        bb_V_nom = jax2np(jax_jit(alg.get_bb_V_nom)())
        bb_V_opt = jax2np(jax_jit(alg.get_bb_V_opt)())
        bb_V_sloped = jax2np(jax_jit(alg.get_bb_V_sloped)())
        bb_V_pred = jax2np(jax_jit(rep_vmap(ft.partial(alg.get_V, alg.V.params), rep=2))(bb_x))

        #####################################################
        logger.info("Plotting...")
        # Plot phase.
        fig, ax = plt.subplots(layout="constrained")
        ax.plot(T_x[:, task.TH], T_x[:, task.W], color="C1", marker="o", ms=0.8, lw=0.5, zorder=4)
        ax.scatter(T_x[0, task.TH], T_x[0, task.W], color="black", s=1**2, zorder=6, marker="s")
        ax.plot(T_x_opt[:, task.TH], T_x_opt[:, task.W], color="C4", marker="o", ms=0.8, lw=0.5, zorder=4)
        ax.plot(T_x_nom[:, task.TH], T_x_nom[:, task.W], color="C3", marker="o", ms=0.8, lw=0.5, zorder=4)

        ax.contour(bb_Xs, bb_Ys, bb_V_pred, levels=[0], colors=["C0"], zorder=3.5)
        ax.contour(bb_Xs, bb_Ys, bb_V_nom, levels=[0], colors=["C5"], zorder=3.5)
        ax.contour(bb_Xs, bb_Ys, bb_V_opt, levels=[0], colors=["C6"], zorder=3.5)
        ax.contour(bb_Xs, bb_Ys, bb_V_sloped, levels=[0], colors=["C2"], zorder=3.5)
        setup.plot(ax)

        lines = [lline("C1"), lline("C4"), lline("C3"), lline("C0"), lline("C5"), lline("C2"), lline("C6")]
        ax.legend(lines, ["QP", "Opt ", "Nominal", "CBF Pred CI", "Nominal CI", "QP CI", "Opt CI"], loc="upper right")

        fig.savefig(plot_dir / f"eval_phase{setup.plot_name}{cid}.pdf")
        # fig.savefig(plot_dir / "eval_phase.png")
        plt.close(fig)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)
