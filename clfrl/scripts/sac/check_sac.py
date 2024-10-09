import ipdb
import matplotlib.pyplot as plt
import numpy as np
import typer
from loguru import logger

import run_config.sac.sac_pend_cfg
import wandb
from clfrl.dyn.pend import Pend
from clfrl.plotting.plot_task_summary import plot_task_summary
from clfrl.plotting.plotter import Plotter
from clfrl.rl.sac import SACAlg
from clfrl.training.ckpt_manager import get_ckpt_manager, save_create_args
from clfrl.training.run_dir import init_wandb_and_get_run_dir
from clfrl.utils.jax_utils import jax2np
from clfrl.utils.logging import set_logger_format


def main(name: str = typer.Option(..., help="Name of the run."), seed: int = 7957821):
    set_logger_format()

    task = Pend()
    # task._dt = 0.2
    # task._dt = 0.025
    # task._dt = 0.01
    # task._dt = 0.05
    task._dt = 0.01

    CFG = run_config.sac.sac_pend_cfg.get(seed)
    sac_alg: SACAlg = SACAlg.create(seed, task, CFG.alg_cfg)

    run_dir = init_wandb_and_get_run_dir(CFG, "clfrl", "sac_pend", name)
    plot_dir, ckpt_dir = run_dir / "plots", run_dir / "ckpts"
    plotter = Plotter(task, plot_dir)

    LCFG = CFG.loop_cfg
    sac_alg, collect_state = sac_alg.init_collect()
    rb = sac_alg.init_rb(LCFG.rb_capacity)

    ckpt_manager = get_ckpt_manager(ckpt_dir, max_to_keep=100)
    save_create_args(ckpt_dir, [seed, task, CFG.alg_cfg])
    plot_task_summary(task, plotter)

    idx_last_log = 0
    sample_rng = np.random.default_rng(seed=515127)
    for idx in range(LCFG.n_iters + 1):
        start_train = idx >= LCFG.start_train
        should_log = idx % LCFG.log_every == 0
        should_eval = idx % LCFG.eval_every == 0
        should_ckpt = idx % LCFG.ckpt_every == 0

        sac_alg, collect_state, b_experience = sac_alg.collect_data(collect_state)
        experience_batch_size = len(b_experience.b_Vobs)
        rb = rb.push_batch(jax2np(b_experience), experience_batch_size)

        is_done = jax2np(collect_state.is_reset_maxtime)
        if idx > 0 and is_done.any() and (idx - idx_last_log) > 100:
            bl_lsum = jax2np(collect_state.bl_lsum)[is_done]
            b_lsum = np.sum(bl_lsum, axis=1)
            logger.info(f"[{idx:8}]   {is_done.sum():03} - lmean: {b_lsum.mean()}")
            idx_last_log = idx

        if not start_train:
            if should_log:
                logger.info(f"[{idx:8}]   ")
            continue

        if idx == LCFG.start_train:
            logger.info("Staring training!")

        batch = rb.uniform_sample_np(sample_rng, sac_alg.epoch_batch_size)
        sac_alg, loss_info = sac_alg.update(batch)

        if should_log:
            log_dict = {f"train/{k}": np.mean(v) for k, v in loss_info.items()}
            log_dict["train/rb_size"] = rb.size
            log_dict["train/collect_idx"] = sac_alg.collect_idx
            logger.info(f"[{idx:8}]   ")
            wandb.log(log_dict, step=idx)
        del loss_info

        if should_eval:
            eval_data: SACAlg.EvalData = jax2np(sac_alg.eval())
            suffix = "{}.jpg".format(idx)

            plotter.batch_phase2d(eval_data.bT_x_plot, f"phase/phase_{suffix}")
            plotter.batch_phase2d(eval_data.bT_x_plot_rng, f"phase_rng/phase_rng_{suffix}")
            plotter.V_seq(eval_data.bb_Xs, eval_data.bb_Ys, eval_data.bbl_Vl_mean[:, :, 0], f"Vl/Vl_{suffix}")

            n_Vl, _, n_e = eval_data.bTe_Vl.shape
            for ii in range(n_Vl):
                fig, axes = plt.subplots(2, dpi=plotter._cfg.dpi, layout="constrained")
                Te_Vl = eval_data.bTe_Vl[ii]

                # Truncate if it stays roughly constant.
                plot_T = 50
                for jj in range(n_e):
                    T_Vl = Te_Vl[:, jj]
                    axes[0].plot(T_Vl[:plot_T], color=f"C{jj}", lw=0.6, alpha=0.75, zorder=4)
                    axes[1].plot(eval_data.bTe_Vl_dot[ii, :plot_T, jj], color=f"C{jj}", lw=0.6, alpha=0.75, zorder=4)

                    # Plot the mean.
                    axes[0].plot(Te_Vl[:plot_T].mean(-1), color="0.8", lw=0.6, alpha=0.5)
                    axes[1].plot(eval_data.bTe_Vl_dot[ii, :plot_T].mean(-1), color="0.8", lw=0.6, alpha=0.5)

                axes[0].set(ylabel=r"$V^l$")
                axes[1].set(ylabel=r"$\dot{V}^l$")
                plotter.savefig(fig, f"Vl_traj/{ii}_{suffix}")

                log_dict = {f"eval/{k}": v.sum() for k, v in eval_data.info.items()}
                wandb.log(log_dict, step=idx)

        if should_ckpt:
            ckpt_manager.save(idx, sac_alg)
            logger.info("[{:5}] - Saving ckpt...".format(idx))

    # Save last.
    logger.info("[{:5}] - Saving ckpt...".format(idx))
    ckpt_manager.save(idx, sac_alg)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)
