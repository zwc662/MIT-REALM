import functools as ft
import pathlib
import pickle

import ipdb
import jax
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import typer
from loguru import logger

import run_config.int_avoid.quadcircle_cfg
import run_config.int_avoid.segway_cfg
import run_config.nclf.pend_cfg
import run_config.nclf_pol.pend_cfg
from pncbf.dyn.quadcircle import QuadCircle
from pncbf.dyn.sim_cts_pbar import SimCtsPbar
from pncbf.ncbf.int_avoid import IntAvoid
from pncbf.plotting.legend_helpers import lline
from pncbf.utils.ckpt_utils import get_id_from_ckpt, get_run_path_from_ckpt, load_ckpt_with_step
from pncbf.utils.jax_utils import jax2np, jax_default_x32, jax_jit, jax_jit_np, jax_vmap, rep_vmap
from pncbf.utils.logging import set_logger_format
from pncbf.utils.path_utils import mkdir


def main(ckpt_path: pathlib.Path, nom_only: bool = False, hocbf: bool = False):
    jax_default_x32()
    set_logger_format()
    task = QuadCircle()

    run_path = get_run_path_from_ckpt(ckpt_path)
    plot_dir = mkdir(run_path / "eval/paper")

    CFG = run_config.int_avoid.quadcircle_cfg.get(0)
    nom_pol = task.nom_pol_vf
    alg: IntAvoid = IntAvoid.create(0, task, CFG.alg_cfg, nom_pol)
    alg, ckpt_path = load_ckpt_with_step(alg, ckpt_path)
    logger.info("Loaded ckpt from {}!".format(ckpt_path))
    cid = get_id_from_ckpt(ckpt_path)

    if nom_only:
        logger.info("Using nom pol")
        pol = nom_pol
    elif hocbf:
        logger.info("Using hocbf")
        pol = task.nom_pol_handcbf
    else:
        pol = ft.partial(alg.get_cbf_control_sloped, 1.0, 50.0)

    tf = 15.0
    dt = task.dt / 2
    n_steps = int(round(tf / dt))

    def get_Vh(x_):
        sim = SimCtsPbar(
            task, pol, n_steps, dt, dt0=dt, use_obs=False, use_pid=False, max_steps=n_steps, solver="bosh3"
        )
        T_x, _ = sim.rollout_plot(x_)
        T_h = jax_vmap(task.h)(T_x)
        hmax = T_h.max()
        return hmax, T_x

    # Sample random x0s that aren't initially unsafe.
    b_x0 = task.sample_stats_x0()
    # Now, eval Vh for all.
    b_Vh, bT_x = jax_jit_np(jax_vmap(get_Vh))(b_x0)

    # Save the stats.
    pkl_path = plot_dir / f"safety_stats{cid}.pkl"
    if nom_only:
        pkl_path = plot_dir / "safety_stats_nom.pkl"
    if hocbf:
        pkl_path = plot_dir / "safety_stats_hocbf.pkl"

    with open(pkl_path, "wb") as f:
        pickle.dump({"Vh": b_Vh, "bT_x": bT_x}, f)
    logger.info("Saved pkl to {}!".format(pkl_path))

    # Print the stats.
    n_safe, safe_frac = np.sum(b_Vh < 0), np.mean(b_Vh < 0)
    logger.info("Safe frac: {:.3f}. {:4} / {:4}".format(safe_frac, n_safe, len(b_Vh)))


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)
