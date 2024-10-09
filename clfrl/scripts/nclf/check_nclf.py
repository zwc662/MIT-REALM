import ipdb
import matplotlib.pyplot as plt
import numpy as np
import typer
from loguru import logger

import run_config.nclf.pend_cfg
import wandb
from clfrl.dyn.pend import Pend
from clfrl.nclf.nclf import NCLF
from clfrl.plotting.plot_task_summary import plot_task_summary
from clfrl.plotting.plotter import Plotter
from clfrl.training.ckpt_manager import get_ckpt_manager, save_create_args
from clfrl.training.run_dir import init_wandb_and_get_run_dir
from clfrl.utils.jax_utils import jax2np
from clfrl.utils.logging import set_logger_format


def main(name: str = typer.Option(..., help="Name of the run."), group: str = typer.Option(None), seed: int = 7957821):
    set_logger_format()

    task = Pend()

    CFG = run_config.nclf.pend_cfg.get(seed)
    alg: NCLF = NCLF.create(seed, task, CFG.alg_cfg)

    run_dir = init_wandb_and_get_run_dir(CFG, "clfrl", "nclf_pend", name, group=group)
    plot_dir, ckpt_dir = run_dir / "plots", run_dir / "ckpts"
    plotter = Plotter(task, plot_dir)

    LCFG = CFG.loop_cfg

    ckpt_manager = get_ckpt_manager(ckpt_dir)
    save_create_args(ckpt_dir, [seed, task, CFG.alg_cfg])
    plot_task_summary(task, plotter)

    # loss_weights = {"Loss/Goal": 10.0, "Loss/V_desc": 1.0, "Loss/Nonzero": 0.1}
    loss_weights = {"Loss/Goal": 10.0, "Loss/V_desc": 1.0, "Loss/Nonzero": 0.0}

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
            eval_data: NCLF.EvalData = jax2np(alg.eval())
            suffix = "{}.jpg".format(idx)

            plotter.batch_phase2d(eval_data.bT_x_plot, f"phase/phase_{suffix}")
            plotter.V_seq(eval_data.bb_Xs, eval_data.bb_Ys, eval_data.bb_V, f"V/V_{suffix}")
            plotter.V_div(eval_data.bb_Xs, eval_data.bb_Ys, eval_data.bb_Vdot, f"Vdot/Vdot_{suffix}")
            plotter.V_div(eval_data.bb_Xs, eval_data.bb_Ys, eval_data.bb_u[:, :, 0], f"u/u_{suffix}")

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
