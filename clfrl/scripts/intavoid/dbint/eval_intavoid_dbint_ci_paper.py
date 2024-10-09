import functools as ft
import pathlib
import pickle

import ipdb
import jax
import numpy as np
import typer
from loguru import logger

import run_config.int_avoid.doubleintwall_cfg
import run_config.int_avoid.segway_cfg
from clfrl.dyn.doubleint_wall import DoubleIntWall
from clfrl.dyn.sim_cts_pbar import SimCtsPbar
from clfrl.ncbf.int_avoid import IntAvoid
from clfrl.utils.ckpt_utils import get_id_from_ckpt, get_run_path_from_ckpt, load_ckpt_with_step
from clfrl.utils.compare_ci import CIData
from clfrl.utils.jax_utils import jax2np, jax_jit, rep_vmap
from clfrl.utils.logging import set_logger_format
from clfrl.utils.path_utils import mkdir


def main(ckpt_path: pathlib.Path):
    set_logger_format()
    task = DoubleIntWall()

    run_path = get_run_path_from_ckpt(ckpt_path)
    plot_dir = mkdir(run_path / "eval")

    CFG = run_config.int_avoid.doubleintwall_cfg.get(0)
    # nom_pol = task.nom_pol_osc
    nom_pol = task.nom_pol_rng2

    alg: IntAvoid = IntAvoid.create(0, task, CFG.alg_cfg, nom_pol)
    alg, ckpt_path = load_ckpt_with_step(alg, ckpt_path)
    logger.info("Loaded ckpt from {}!".format(ckpt_path))
    cid = get_id_from_ckpt(ckpt_path)

    pol = ft.partial(alg.get_cbf_control_sloped, 2.0, 100.0)

    tf = 8.0
    dt = 0.05
    n_steps = int(round(tf / dt))
    logger.info("n_steps: {}".format(n_steps))

    def get_bb_rollout(bb_x_):
        sim = SimCtsPbar(
            task, pol, n_steps, dt, dt0=dt, use_obs=False, use_pid=False, max_steps=n_steps, solver="bosh3"
        )
        bbT_x, _ = rep_vmap(sim.rollout_plot, rep=2)(bb_x_)
        return bbT_x

    get_bb_rollout = jax.jit(get_bb_rollout)

    setup_idx = 0
    bb_x, bb_Xs, bb_Ys = task.get_paper_ci_x0()
    bbT_x = jax2np(jax_jit(get_bb_rollout)(bb_x))
    bbTh_h = jax2np(jax_jit(rep_vmap(task.h_components, rep=3))(bbT_x))
    bbTh_Vh = jax2np(jax_jit(rep_vmap(alg.get_Vh, rep=3))(bbT_x))

    bbT_u = np.zeros(0)

    ci_data = CIData("IntAvoid", task.name, setup_idx, bbT_x, bbT_u, bbTh_h, bb_Xs, bb_Ys, notes={"bbTh_Vh": bbTh_Vh})
    pkl_path = pathlib.Path(plot_dir / f"dbint{cid}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(ci_data, f)

    logger.info("Saved to {}!".format(pkl_path))


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)
