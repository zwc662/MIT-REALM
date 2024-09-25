import functools as ft
import pathlib

import ipdb
import numpy as np
import typer
from loguru import logger

import run_config.avoid_fixed.doubleintwall_cfg
import wandb
from pncbf.dyn.doubleint_wall import DoubleIntWall
from pncbf.ncbf.avoid_fixed import AvoidFixed
from pncbf.plotting.plot_task_summary import plot_task_summary
from pncbf.plotting.plotter import Plotter
from pncbf.training.ckpt_manager import get_ckpt_manager, save_create_args
from pncbf.training.run_dir import init_wandb_and_get_run_dir
from pncbf.utils.ckpt_utils import load_ckpt
from pncbf.utils.jax_utils import jax2np, jax_jit
from pncbf.utils.logging import set_logger_format


def main(
    ckpt_path: pathlib.Path,
    name: str = typer.Option(..., help="Name of the run."),
    group: str = typer.Option(None),
    seed: int = 7957821,
):
    set_logger_format()

    task = DoubleIntWall()
    CFG = run_config.avoid_fixed.doubleintwall_cfg.get(seed)
    alg_pre: AvoidFixed = AvoidFixed.create(seed, task, CFG.alg_cfg, task.nom_pol_rng)
    alg_pre = load_ckpt(alg_pre, ckpt_path)
    logger.info("Loaded ckpt from {}!".format(ckpt_path))

    # Another instance, but using the policy from the first one.
    nom_pol = alg_pre.get_opt_u
    # CFG.alg_cfg.train_cfg.lam = 1.0
    CFG.extras = {"nom_pol": "pretrained"}
    alg: AvoidFixed = AvoidFixed.create(seed, task, CFG.alg_cfg, nom_pol)

    loss_weights = {"Loss/Now": 10.0, "Loss/Future": 10.0, "Loss/PDE": 0.1}
    CFG.extras["loss_weights"] = loss_weights

    run_dir = init_wandb_and_get_run_dir(CFG, "pncbf", "avoidfixed_doubleintwall_pi", name, group=group)
    plot_dir, ckpt_dir = run_dir / "plots", run_dir / "ckpts"
    plotter = Plotter(task, plot_dir)

    LCFG = CFG.loop_cfg

    ckpt_manager = get_ckpt_manager(ckpt_dir)
    save_create_args(ckpt_dir, [seed, task, CFG.alg_cfg])
    plot_task_summary(task, plotter, nom_pol=nom_pol)

    bb_V_nom = jax2np(jax_jit(alg.get_bb_V_nom)())
    V_nom_line = [(bb_V_nom, "C5")]

    for idx in range(LCFG.n_iters + 1):
        should_log = idx % LCFG.log_every == 0
        should_eval = idx % LCFG.eval_every == 0
        should_ckpt = idx % LCFG.ckpt_every == 0

        alg, batch = alg.sample_batch()
        alg, loss_info = alg.update(batch, loss_weights)

        if should_log:
            log_dict = {f"train/{k}": np.mean(v) for k, v in loss_info.items()}
            log_dict["train/collect_idx"] = alg.collect_idx
            logger.info(f"[{idx:8}]   ")
            wandb.log(log_dict, step=idx)
        del loss_info

        if should_eval:
            eval_data: AvoidFixed.EvalData = jax2np(alg.eval())
            suffix = "{}.jpg".format(idx)

            bb_Xs, bb_Ys = eval_data.bb_Xs, eval_data.bb_Ys
            plotter.batch_phase2d(eval_data.bT_x_plot, f"phase/phase_{suffix}")
            plotter.V_div(bb_Xs, bb_Ys, eval_data.bbh_V.max(-1), f"V/V_{suffix}", extra_lines=V_nom_line)
            plotter.V_div(bb_Xs, bb_Ys, eval_data.bbh_Vdot.max(-1), f"Vdot_max/Vdot_max_{suffix}")
            plotter.V_div(bb_Xs, bb_Ys, eval_data.bbh_Vdot_disc.max(-1), f"Vdot_disc_max/Vdot_disc_max_{suffix}")
            plotter.V_div(bb_Xs, bb_Ys, eval_data.bbh_hmV.max(-1), f"hmV_max/hmV_max_{suffix}", extra_lines=V_nom_line)
            plotter.V_div(bb_Xs, bb_Ys, eval_data.bbh_compl.max(-1), f"compl_max/compl_max_{suffix}", extra_lines=V_nom_line)
            # plotter.V_div(bb_Xs, bb_Ys, 1.0 * eval_data.bb_eq_h, f"eq_h/eq_h_{suffix}")
            plotter.V_div(bb_Xs, bb_Ys, eval_data.bb_u[:, :, 0], f"u/u_{suffix}")

            log_dict = {f"eval/{k}": v.sum() for k, v in eval_data.info.items()}
            wandb.log(log_dict, step=idx)

        if should_ckpt:
            ckpt_manager.save(idx, alg)
            logger.info("[{:5}] - Saving ckpt...".format(idx))

    # Save last.
    logger.info("[{:5}] - Saving ckpt...".format(idx))
    ckpt_manager.save(idx, alg, force=True)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)
