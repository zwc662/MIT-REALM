import ipdb
import numpy as np
import typer
from flax.training import orbax_utils
from loguru import logger

import run_config.ncbf.quadcircle_cfg
import wandb
from clfrl.dyn.quadcircle import QuadCircle
from clfrl.ncbf.ncbf import NCBF
from clfrl.plotting.plot_task_summary import plot_task_summary
from clfrl.plotting.plotter import MPPlotter, Plotter
from clfrl.training.ckpt_manager import get_ckpt_manager, save_create_args
from clfrl.training.run_dir import init_wandb_and_get_run_dir
from clfrl.utils.jax_utils import jax2np, jax_jit, tree_cat, tree_copy
from clfrl.utils.logging import set_logger_format


def main(name: str = typer.Option(..., help="Name of the run."), group: str = typer.Option(None), seed: int = 7957821):
    set_logger_format()

    task = QuadCircle()

    CFG = run_config.ncbf.quadcircle_cfg.get(seed)

    nom_pol = task.nom_pol_vf

    alg: NCBF = NCBF.create(seed, task, CFG.alg_cfg, nom_pol)
    CFG.extras = {"nom_pol": "lqr", "buffer_assert": task.buffer_assert}

    loss_weights = {"Loss/Unsafe": 10.0, "Loss/Safe": 10.0, "Loss/Descent": 0.01}
    CFG.extras["loss_weights"] = loss_weights

    run_dir = init_wandb_and_get_run_dir(CFG, "ncbf", "ncbf_quadcircle", name, group=group)
    plot_dir, ckpt_dir = run_dir / "plots", run_dir / "ckpts"
    plotter = MPPlotter(task, plot_dir)

    LCFG = CFG.loop_cfg

    ckpt_manager = get_ckpt_manager(ckpt_dir)
    save_create_args(ckpt_dir, [seed, task, CFG.alg_cfg])
    plot_task_summary(task, plotter, nom_pol=nom_pol)

    _, bb_Xs, bb_Ys = task.get_contour_x0()
    del _

    bb_V_noms = jax2np(task.get_bb_V_noms())
    Vnom_ln = [(bb_V_noms, "C5")]

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
            eval_data: NCBF.EvalData = jax2np(alg.eval())
            suffix = "{}.jpg".format(idx)

            bb_Xs, bb_Ys = eval_data.bb_Xs, eval_data.bb_Ys

            plotter.mp_run.remove_finished()
            plotter.batch_phase2d(eval_data.bT_x_plot, f"phase/phase_{suffix}", extra_lines=Vnom_ln)
            plotter.V_div(bb_Xs, bb_Ys, eval_data.bb_V, f"V/V_{suffix}", extra_lines=Vnom_ln)
            plotter.V_div(bb_Xs, bb_Ys, eval_data.bb_Vdot, f"dV_max/dV_{suffix}", extra_lines=Vnom_ln)
            # plotter.V_div(bb_Xs, bb_Ys, eval_data.bbh_hmV.max(-1), f"hmV_max/hmV_max_{suffix}", extra_lines=V_nom_line)
            # plotter.V_div(
            #     bb_Xs, bb_Ys, eval_data.bbh_compl.max(-1), f"compl_max/compl_max_{suffix}", extra_lines=V_nom_line
            # )
            # plotter.V_div(bb_Xs, bb_Ys, eval_data.bb_u[:, :, 0], f"u/u_{suffix}")

            log_dict = {f"eval/{k}": v.sum() for k, v in eval_data.info.items()}
            wandb.log(log_dict, step=idx)

        if should_ckpt:
            alg_copy = tree_copy(alg)
            save_args = orbax_utils.save_args_from_target(alg_copy)
            ckpt_manager.save(idx, alg_copy, save_kwargs={"save_args": save_args})
            logger.info("[{:5}] - Saving ckpt...".format(idx))

    # Save last.
    logger.info("[{:5}] - Saving ckpt...".format(idx))
    save_args = orbax_utils.save_args_from_target(alg)
    ckpt_manager.save(idx, alg, save_kwargs={"save_args": save_args})


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)
