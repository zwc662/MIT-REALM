import functools as ft
import pathlib
import pickle

import ipdb
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import typer
from loguru import logger
from matplotlib.animation import FuncAnimation

import clfrl.ncbf.min_norm_cbf as cbf_old
import run_config.int_avoid.quadcircle_cfg
import run_config.int_avoid.segway_cfg
import run_config.nclf.pend_cfg
import run_config.nclf_pol.pend_cfg
from clfrl.dyn.quadcircle import QuadCircle
from clfrl.dyn.sim_cts import SimCtsReal
from clfrl.dyn.sim_cts_pbar import SimCtsPbar
from clfrl.ncbf.compute_disc_avoid import compute_all_disc_avoid_terms
from clfrl.ncbf.int_avoid import IntAvoid
from clfrl.plotting.ani_utils import save_anim
from clfrl.plotting.legend_helpers import lline
from clfrl.plotting.plotter import pretty_str
from clfrl.qp.min_norm_cbf import min_norm_cbf
from clfrl.utils.ckpt_utils import get_id_from_ckpt, get_run_path_from_ckpt, load_ckpt, load_ckpt_with_step
from clfrl.utils.easy_npz import EasyNpz
from clfrl.utils.jax_utils import jax2np, jax_jit, jax_vmap, rep_vmap
from clfrl.utils.logging import set_logger_format
from clfrl.utils.path_utils import mkdir

app = typer.Typer()


@app.command()
def gen(ckpt_path: pathlib.Path):
    set_logger_format()
    task = QuadCircle()

    run_path = get_run_path_from_ckpt(ckpt_path)
    plot_dir = mkdir(run_path / "eval")

    CFG = run_config.int_avoid.quadcircle_cfg.get(0)
    nom_pol = task.nom_pol_vf
    alg: IntAvoid = IntAvoid.create(0, task, CFG.alg_cfg, nom_pol)
    alg, ckpt_path = load_ckpt_with_step(alg, ckpt_path)
    logger.info("Loaded ckpt from {}!".format(ckpt_path))
    cid = get_id_from_ckpt(ckpt_path)

    pols = {
        "Nom": nom_pol,
        "HOCBF": task.nom_pol_handcbf,
        "NCBF": ft.partial(alg.get_cbf_control_sloped, 2.0, 100.0),
    }

    # tf = 10.0
    tf = 8.0
    dt = 0.05
    n_pts = 128
    n_steps = int(round(tf / dt))
    tf = n_steps * dt
    logger.info("n_steps: {}".format(n_steps))

    def get_bb_Vh(pol, bb_x_):
        # sim = SimCtsReal(task, pol, tf, dt, dt0=dt, use_obs=False, use_pid=False, max_steps=n_steps, solver="bosh3")
        sim = SimCtsPbar(
            task, pol, n_steps, dt, dt0=dt, use_obs=False, use_pid=False, max_steps=n_steps, solver="bosh3"
        )
        bbT_x, _ = rep_vmap(sim.rollout_plot, rep=2)(bb_x_)
        bbT_h = rep_vmap(task.h, rep=3)(bbT_x)
        bb_h = bbT_h.max(axis=-1)
        return bb_h

    get_bb_Vh_pol = {k: jax_jit(ft.partial(get_bb_Vh, pol)) for k, pol in pols.items()}

    vmap_jit_get_V = jax.jit(rep_vmap(alg.get_V, rep=2))

    ###########################################################
    bb_Vh_dicts = []
    for setup in task.phase2d_setups():
        logger.info(f"Generating {setup.plot_name}...")

        bb_x, bb_Xs, bb_Ys = task.get_ci_x0(setup=setup.setup_idx, n_pts=n_pts)

        bb_Vh_dict = {}
        for pol_name, pol in pols.items():
            logger.info(f"    {pol_name}...")
            bb_h = jax2np(get_bb_Vh_pol[pol_name](bb_x))
            bb_Vh_dict[pol_name] = bb_h

        bb_Vh_dicts.append(bb_Vh_dict)

        # Also add in the predicted CI
        bb_Vh_dict["Pred"] = jax2np(vmap_jit_get_V(bb_x))

    pkl_path = plot_dir / f"cidata{cid}.pkl"
    with open(pkl_path, "wb") as f:
        data = [bb_Vh_dicts, n_pts]
        pickle.dump(data, f)

    plot(ckpt_path)


@app.command()
def plot(ckpt_path: pathlib.Path):
    set_logger_format()
    task = QuadCircle()

    run_path = get_run_path_from_ckpt(ckpt_path)
    plot_dir = mkdir(run_path / "eval")

    cid = get_id_from_ckpt(ckpt_path)

    ###########################################################
    pkl_path = plot_dir / f"cidata{cid}.pkl"
    logger.info("Loading from {}...".format(pkl_path))
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    logger.info("Loaded!")
    bb_Vh_dicts, n_pts = data
    ###########################################################

    colors = {"HOCBF": "C0", "NCBF": "C1", "Nom": "C5", "Pred": "C4"}
    levels = [0]
    linestyles = ["-"]

    ###########################################################
    for setup in task.phase2d_setups():
        logger.info(f"Plotting {setup.plot_name}...")

        bb_Vh_dict = bb_Vh_dicts[setup.setup_idx]

        # bb_x, bb_Xs, bb_Ys = task.get_contour_x0(setup.setup_idx, n_pts=n_pts)
        bb_x, bb_Xs, bb_Ys = task.get_ci_x0(setup.setup_idx, n_pts=n_pts)
        fig, ax = plt.subplots(layout="constrained")
        for pol_name, bb_Vh in bb_Vh_dict.items():
            color = colors[pol_name]
            ax.contourf(bb_Xs, bb_Ys, np.ma.array(bb_Vh <= 0, mask=bb_Vh > 0), colors=[color], alpha=0.2, zorder=3.5)
            ax.contour(bb_Xs, bb_Ys, bb_Vh, levels=levels, linestyles=linestyles, colors=[color], zorder=3.5)
        setup.plot(ax)

        lines = [lline(color) for color in colors.values()]
        labels = list(colors.keys())
        ax.legend(lines, labels, loc="upper right")
        fig.savefig(plot_dir / f"ci_{setup.plot_name}{cid}.pdf")
        plt.close(fig)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
