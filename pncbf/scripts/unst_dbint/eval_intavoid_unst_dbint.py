import functools as ft
import pathlib

import ipdb
import matplotlib.pyplot as plt
import typer
from loguru import logger
from matplotlib.colors import CenteredNorm

import run_config.int_avoid.unst_dbint_cfg
import run_config.nclf.pend_cfg
import run_config.nclf_pol.pend_cfg
from pncbf.dyn.sim_cts_pbar import SimCtsPbar
from pncbf.dyn.unst_dbint import UnstDbInt
from pncbf.ncbf.int_avoid import IntAvoid
from pncbf.plotting.legend_helpers import lline
from pncbf.plotting.plotstyle import PlotStyle
from pncbf.utils.ckpt_utils import get_id_from_ckpt, get_run_path_from_ckpt, load_ckpt
from pncbf.utils.jax_utils import jax2np, jax_default_x32, jax_jit, rep_vmap
from pncbf.utils.logging import set_logger_format
from pncbf.utils.path_utils import mkdir


def main(ckpt_path: pathlib.Path):
    jax_default_x32()
    set_logger_format()
    seed = 0

    run_path = get_run_path_from_ckpt(ckpt_path)
    plot_dir = mkdir(run_path / "eval")

    task = UnstDbInt()

    nom_pol = task.nom_pol_zero

    CFG = run_config.int_avoid.unst_dbint_cfg.get(seed)
    alg: IntAvoid = IntAvoid.create(seed, task, CFG.alg_cfg, nom_pol)
    alg = load_ckpt(alg, ckpt_path)
    logger.info("Loaded ckpt from {}!".format(ckpt_path))
    cid = get_id_from_ckpt(ckpt_path)

    pol = ft.partial(alg.get_cbf_control_sloped, 2.0, 100.0)

    n_steps = 150
    dt = task.dt

    def get_bb_Vh(bb_x_):
        sim = SimCtsPbar(
            task, pol, n_steps, dt, dt0=dt, use_obs=False, use_pid=False, max_steps=n_steps, solver="tsit5"
        )
        bbT_x, _ = rep_vmap(sim.rollout_plot, rep=2)(bb_x_)
        bbT_h = rep_vmap(task.h, rep=3)(bbT_x)
        bb_h = bbT_h.max(axis=-1)
        return bb_h

    bb_x, bb_Xs, bb_Ys = task.get_contour_x0(setup=0)
    bb_Vh = jax2np(jax_jit(get_bb_Vh)(bb_x))
    bb_Vh_pred = jax2np(jax_jit(rep_vmap(alg.get_V, rep=2))(bb_x))
    ############################################################################
    fig, ax = plt.subplots(layout="constrained")
    task.plot_phase(ax)
    cs0 = ax.contourf(bb_Xs, bb_Ys, bb_Vh, levels=11, norm=CenteredNorm(), cmap="RdBu_r")
    cs1 = ax.contour(bb_Xs, bb_Ys, bb_Vh, levels=[0.0], colors=[PlotStyle.ZeroColor], alpha=0.98, linewidths=1.0)
    cs2 = ax.contour(bb_Xs, bb_Ys, bb_Vh_pred, levels=[0.0], colors=["C4"], alpha=0.98, linewidths=1.0)
    cbar = fig.colorbar(cs0, shrink=0.9)
    cbar.add_lines(cs1)
    lines = [lline(PlotStyle.ZeroColor), lline("C4")]
    labels = ["True", "Pred"]
    ax.legend(lines, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 1.0), fontsize="x-small")
    fig.savefig(plot_dir / f"ci{cid}.pdf")


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)
