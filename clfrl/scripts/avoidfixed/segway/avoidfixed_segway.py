import functools as ft

import ipdb
import numpy as np
import typer
from loguru import logger

import run_config.avoid_fixed.segway_cfg
import wandb
from clfrl.dyn.segway import Segway
from clfrl.ncbf.avoid_fixed import AvoidFixed
from clfrl.plotting.plot_task_summary import plot_task_summary
from clfrl.plotting.plotter import Plotter, MPPlotter
from clfrl.training.ckpt_manager import get_ckpt_manager, save_create_args
from clfrl.training.run_dir import init_wandb_and_get_run_dir
from clfrl.utils.jax_utils import jax2np, jax_jit, rep_vmap
from clfrl.utils.logging import set_logger_format


def main(name: str = typer.Option(..., help="Name of the run."), group: str = typer.Option(None), seed: int = 7957821):
    set_logger_format()

    task = Segway()

    CFG = run_config.avoid_fixed.segway_cfg.get(seed)
    nom_T = 500

    nom_pol = task.nom_pol_lqr

    alg: AvoidFixed = AvoidFixed.create(seed, task, CFG.alg_cfg, nom_pol)
    CFG.extras = {"nom_pol": "lqr"}

    # loss_weights = {"Loss/Now": 10.0, "Loss/Future": 10.0, "Loss/PDE": 0.1}
    # loss_weights = {"Loss/Now": 10.0, "Loss/Future": 10.0, "Loss/PDE": 0.01}
    # loss_weights = {"Loss/Now": 10.0, "Loss/Future": 10.0, "Loss/PDE": 1.0}
    loss_weights = {"Loss/Now": 2.0, "Loss/Future": 5.0, "Loss/PDE": 2.0}
    CFG.extras["loss_weights"] = loss_weights

    run_dir = init_wandb_and_get_run_dir(CFG, "clfrl", "avoidfixed_segway", name, group=group)
    plot_dir, ckpt_dir = run_dir / "plots", run_dir / "ckpts"
    plotter = MPPlotter(task, plot_dir)

    LCFG = CFG.loop_cfg

    ckpt_manager = get_ckpt_manager(ckpt_dir)
    save_create_args(ckpt_dir, [seed, task, CFG.alg_cfg])
    plot_task_summary(task, plotter, nom_pol=nom_pol)

    bb_V_nom = jax2np(alg.get_bb_V_nom(nom_T))
    V_nom_line = [(bb_V_nom, "C5")]

    for idx in range(LCFG.n_iters + 1):
        should_log = idx % LCFG.log_every == 0
        should_eval = idx % LCFG.eval_every == 0
        should_plot = idx % LCFG.plot_every == 0
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

            if should_plot:
                suffix = "{}.jpg".format(idx)
                bb_Xs, bb_Ys = eval_data.bb_Xs, eval_data.bb_Ys
                plotter.mp_run.remove_finished()
                plotter.batch_phase2d(
                    eval_data.bT_x_plot, f"phase/phase_{suffix}", extra_lines=[(bb_Xs, bb_Ys, bb_V_nom, "C5")]
                )
                plotter.V_div(bb_Xs, bb_Ys, eval_data.bbh_V.max(-1), f"V/V_{suffix}", extra_lines=V_nom_line)
                plotter.V_div(bb_Xs, bb_Ys, eval_data.bbh_Vdot.max(-1), f"Vdot_max/Vdot_max_{suffix}")
                plotter.V_div(bb_Xs, bb_Ys, eval_data.bbh_Vdot_disc.max(-1), f"Vdot_disc_max/Vdot_disc_max_{suffix}")
                plotter.V_div(
                    bb_Xs, bb_Ys, eval_data.bbh_hmV.max(-1), f"hmV_max/hmV_max_{suffix}", extra_lines=V_nom_line
                )
                plotter.V_div(
                    bb_Xs, bb_Ys, eval_data.bbh_compl.max(-1), f"compl_max/compl_max_{suffix}", extra_lines=V_nom_line
                )
                # plotter.V_div(bb_Xs, bb_Ys, 1.0 * eval_data.bb_eq_h, f"eq_h/eq_h_{suffix}")
                plotter.V_div(bb_Xs, bb_Ys, eval_data.bb_u[:, :, 0], f"u/u_{suffix}")

            log_dict = {f"eval/{k}": v.sum() for k, v in eval_data.info.items()}
            wandb.log(log_dict, step=idx)

        if should_ckpt:
            ckpt_manager.save(idx, alg)
            logger.info("[{:5}] - Saving ckpt...".format(idx))

    # Save last.
    logger.info("[{:5}] - Saving ckpt...".format(idx))
    ckpt_manager.save(idx, alg)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)
