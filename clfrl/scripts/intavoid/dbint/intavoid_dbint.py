import time

import ipdb
import numpy as np
import typer
from loguru import logger

import run_config.int_avoid.doubleintwall_cfg
import wandb
from clfrl.dyn.doubleint_wall import DoubleIntWall
from clfrl.ncbf.int_avoid import IntAvoid
from clfrl.plotting.plot_task_summary import plot_task_summary
from clfrl.plotting.plotter import MPPlotter, Plotter
from clfrl.training.ckpt_manager import get_ckpt_manager, save_create_args
from clfrl.training.run_dir import init_wandb_and_get_run_dir
from clfrl.utils.jax_utils import jax2np, jax_jit, tree_cat
from clfrl.utils.logging import set_logger_format


def main(name: str = typer.Option(..., help="Name of the run."), group: str = typer.Option(None), seed: int = 7957821):
    set_logger_format()

    task = DoubleIntWall()

    CFG = run_config.int_avoid.doubleintwall_cfg.get(seed)

    # nom_pol = task.nom_pol_rng
    # nom_pol = task.nom_pol_rng2
    nom_pol = task.nom_pol_osc

    # alg: AvoidFixed = AvoidFixed.create(seed, task, CFG.alg_cfg, task.nom_pol_osc)
    alg: IntAvoid = IntAvoid.create(seed, task, CFG.alg_cfg, nom_pol)
    CFG.extras = {"nom_pol": "pp"}

    # loss_weights = {"Loss/Vh_mse": 1.0, "Loss/Now": 0.0, "Loss/Future": 0.1, "Loss/PDE": 0.0}
    loss_weights = {"Loss/Vh_mse": 1.0, "Loss/Now": 1.0, "Loss/Future": 1.0, "Loss/PDE": 0.0}
    CFG.extras["loss_weights"] = loss_weights

    run_dir = init_wandb_and_get_run_dir(CFG, "intavoid_dbint", "intavoid_dbint", name, group=group)
    plot_dir, ckpt_dir = run_dir / "plots", run_dir / "ckpts"
    plotter = MPPlotter(task, plot_dir)

    LCFG = CFG.loop_cfg

    ckpt_manager = get_ckpt_manager(ckpt_dir)
    save_create_args(ckpt_dir, [seed, task, CFG.alg_cfg])
    plot_task_summary(task, plotter, nom_pol=nom_pol)

    _, bb_Xs, bb_Ys = task.get_contour_x0()
    del _

    bb_V_nom = jax2np(alg.get_bb_V_nom())
    V_nom_line = [(bb_V_nom, "C5")]
    V_nom_line2 = [(bb_Xs, bb_Ys, bb_V_nom, "C5")]

    dset_list = []
    dset_len_max = 8

    rng = np.random.default_rng(seed=58123)
    dset = None
    for idx in range(LCFG.n_iters + 1):
        should_log = idx % LCFG.log_every == 0
        should_eval = idx % LCFG.eval_every == 0
        should_ckpt = idx % LCFG.ckpt_every == 0

        if (idx % 500 == 0) or len(dset_list) < dset_len_max:
            alg, dset = alg.sample_dset()
            dset_list.append(jax2np(dset))

            if len(dset_list) > dset_len_max:
                del dset_list[0]

            dset = tree_cat(dset_list, axis=0)
            b = dset.bT_x.shape[0]
            b_times_Tm1 = b * (dset.bT_x.shape[1] - 1)

        # Randomly sample x0. Half is random, half is t=0.
        n_rng = alg.train_cfg.batch_size // 2
        n_zero = alg.train_cfg.batch_size - n_rng

        b_idx_rng = rng.integers(0, b_times_Tm1, size=(n_rng,))
        b_idx_b_rng = b_idx_rng // (dset.bT_x.shape[1] - 1)
        b_idx_t_rng = 1 + (b_idx_rng % (dset.bT_x.shape[1] - 1))

        b_idx_b_zero = rng.integers(0, b, size=(n_zero,))
        b_idx_t_zero = np.zeros_like(b_idx_b_zero)

        b_idx_b = np.concatenate([b_idx_b_rng, b_idx_b_zero], axis=0)
        b_idx_t = np.concatenate([b_idx_t_rng, b_idx_t_zero], axis=0)

        b_x0 = dset.bT_x[b_idx_b, b_idx_t]
        b_xT = dset.bT_x[b_idx_b, -1]
        bh_lhs = dset.b_vterms.Th_max_lhs[b_idx_b, b_idx_t, :]
        bh_int_rhs = dset.b_vterms.Th_disc_int_rhs[b_idx_b, b_idx_t, :]
        b_discount_rhs = dset.b_vterms.T_discount_rhs[b_idx_b, b_idx_t]
        batch = alg.Batch(b_x0, b_xT, bh_lhs, bh_int_rhs, b_discount_rhs)

        alg, loss_info = alg.update(batch, loss_weights)

        if should_log:
            log_dict = {f"train/{k}": np.mean(v) for k, v in loss_info.items()}
            log_dict["train/collect_idx"] = alg.collect_idx
            logger.info(f"[{idx:8}]   ")
            wandb.log(log_dict, step=idx)

        del loss_info

        if should_eval:
            logger.info("Evaluating...")
            d: IntAvoid.EvalData = jax2np(alg.eval())
            logger.info("Evaluating... Done!")
            suffix = "{}.jpg".format(idx)

            bb_Xs, bb_Ys = d.bb_Xs, d.bb_Ys
            bb_Vdot_disc_max = d.bbh_Vdot_disc.max(-1)

            plotter.mp_run.remove_finished()
            logger.info("Plotting...")
            plotter.batch_phase2d(d.bT_x_plot, f"phase/phase_{suffix}", extra_lines=V_nom_line2)
            plotter.V_div(bb_Xs, bb_Ys, d.bbh_V.max(-1), f"V/V_{suffix}", extra_lines=V_nom_line)
            plotter.V_div(bb_Xs, bb_Ys, d.bbh_Vdot.max(-1), f"dV/dV_{suffix}", extra_lines=V_nom_line)
            plotter.V_div(bb_Xs, bb_Ys, bb_Vdot_disc_max, f"dV_disc/dV_disc_{suffix}", extra_lines=V_nom_line)
            # plotter.V_div(bb_Xs, bb_Ys, eval_data.bbh_hmV.max(-1), f"hmV_max/hmV_max_{suffix}", extra_lines=V_nom_line)
            # plotter.V_div(
            #     bb_Xs, bb_Ys, eval_data.bbh_compl.max(-1), f"compl_max/compl_max_{suffix}", extra_lines=V_nom_line
            # )
            # plotter.V_div(bb_Xs, bb_Ys, eval_data.bb_u[:, :, 0], f"u/u_{suffix}")
            logger.info("Plotting... Done!")

            log_dict = {f"eval/{k}": v.sum() for k, v in d.info.items()}
            wandb.log(log_dict, step=idx)

        if should_ckpt:
            logger.info("[{:5}] - Saving ckpt...".format(idx))
            ckpt_manager.save(idx, alg)
            logger.info("[{:5}] - Saving ckpt... Done!".format(idx))

    # Save last.
    logger.info("[{:5}] - Saving ckpt...".format(idx))
    ckpt_manager.save(idx, alg)
    ckpt_manager.wait_until_finished()


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)
