import pathlib

import ipdb
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import typer
from loguru import logger

import run_config.nclffixed.pend_cfg
import run_config.sac.sac_pend_cfg
import wandb
from clfrl.dyn.pend import Pend
from clfrl.dyn.sim_cts import SimCts
from clfrl.nclf.nclf_fixed import NCLFFixed
from clfrl.plotting.plot_task_summary import plot_task_summary
from clfrl.plotting.plotter import Plotter
from clfrl.rl.sac import SACAlg
from clfrl.training.ckpt_manager import get_ckpt_manager, load_create_args, save_create_args
from clfrl.training.run_dir import init_wandb_and_get_run_dir
from clfrl.utils.ckpt_utils import load_ckpt
from clfrl.utils.jax_utils import jax2np, jax_jit, jax_vmap, rep_vmap
from clfrl.utils.logging import set_logger_format
from clfrl.utils.path_utils import mkdir
from clfrl.utils.paths import get_plots_dir, get_root_dir, get_runs_dir


def main():
    root_ckpt_path = get_runs_dir() / "sac_pend"
    ckpt_dirs = {
        0.01: "2023-07-27_13-11_RIRY_takai_tanh_dt=0.01",
        0.05: "2023-07-27_13-06_TYFF_takai_tanh_dt=0.05",
        0.1: "2023-07-27_12-56_IJHJ_takai_tanh_dt=0.1",
        0.2: "2023-07-27_13-02_ETDE_takai_tanh_dt=0.2",
    }
    # ckpt_num = 15_000
    ckpt_num = 35_000
    ckpt_k = ckpt_num // 1_000
    ckpt_paths = {k: root_ckpt_path / v / f"ckpts/{ckpt_num}" for k, v in ckpt_dirs.items()}

    task = Pend()

    CFG_SAC = run_config.sac.sac_pend_cfg.get(0)
    sac: SACAlg = SACAlg.create(0, task, CFG_SAC.alg_cfg)
    pols = {}
    for k, v in ckpt_paths.items():
        logger.info("Loading {}...".format(v))
        sac = load_ckpt(sac, v)
        pols[k] = sac.policy

    total_s = 4.0
    pol_dt = 0.2
    # pol_dt = 0.1
    # pol_dt = 0.01
    # pol_dt = 0.005
    T = int(round(total_s / pol_dt))

    interp_pts = max(1, int(round(500 / T)))

    # b_x0 = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    key = jr.PRNGKey(8972345)
    batch_size = 128
    xmax = np.array([0.5, 5.0])
    xmin = -xmax
    b_x0 = jr.uniform(key, (batch_size, 2), minval=xmin, maxval=xmax)

    apply_fn = pols[0.1].apply_fn

    @jax_jit
    def sim_pol(params):
        pol_mode = lambda obs: apply_fn(params, obs).mode()
        sim = SimCts(task, pol_mode, T, interp_pts, pol_dt=pol_dt)
        bT_state, _, bT_t = jax_vmap(sim.rollout_plot)(b_x0)
        bT_l = rep_vmap(task.l_cts, rep=2)(bT_state)
        return bT_state, bT_l, bT_t[0]

    bT_states, bT_ls = {}, {}
    for k, p in pols.items():
        logger.info("Simulating {}...".format(k))
        bT_states[k], bT_ls[k], T_ts = jax2np(sim_pol(p.params))

    plot_dir = mkdir(get_plots_dir() / "eval_sac_dt")

    XLIM = np.array([-1.2 * np.pi, 1.2 * np.pi])
    YLIM = np.array([-20.0, 20.0])

    ls = ["-", "--", "-."]
    n_plot = len(ls)

    ##################################################################
    fig, ax = plt.subplots(layout="constrained")
    for ii, (train_dt, bT_states) in enumerate(bT_states.items()):
        for jj, T_states in enumerate(bT_states[:n_plot]):
            ax.plot(T_states[:, 0], T_states[:, 1], color=f"C{ii + 1}", lw=0.8, ls=ls[jj], alpha=0.7, zorder=3)
        ax.plot([0], [0], color=f"C{ii + 1}", label="{:.2f}".format(train_dt))
    ax.legend()
    # Horizontal and vertical line at the goal.
    ax.axvline(np.pi)
    ax.axvline(-np.pi)
    ax.axhline(0.0)
    ax.set(xlim=XLIM, ylim=YLIM)
    fig.savefig(plot_dir / "phase__pol_dt={}_c{}k.pdf".format(pol_dt, ckpt_k))
    plt.close(fig)

    ##################################################################
    # Also plot evolution of costs for quick summary.
    # ts = np.linspace(0, T * pol_dt, num=T_states.shape[0])
    T_ts

    fig, ax = plt.subplots(layout="constrained")
    for ii, (train_dt, bT_l) in enumerate(bT_ls.items()):
        for jj, T_l in enumerate(bT_l[:n_plot]):
            ax.plot(T_ts, T_l, color=f"C{ii + 1}", lw=0.8, ls=ls[jj], alpha=0.5, zorder=3)
        ax.plot([0], [0], color=f"C{ii + 1}", label="{:.2f}".format(train_dt))
    ax.legend()
    fig.savefig(plot_dir / "cost__pol_dt={}_c{}k.pdf".format(pol_dt, ckpt_k))
    plt.close(fig)

    ##################################################################
    # Plot the distribution of limsup. Take the last 1 second.
    T_is_final = T_ts >= (T_ts[-1] - 1.0)
    # b_limsups = {}
    # for k, bT_l in bT_ls.items():
    #     b_limsups[k] = bT_l[is_final].max()
    b_limsups = [bT_l[:, T_is_final].max(axis=-1) for bT_l in bT_ls.values()]

    labels = list(bT_ls.keys())
    ax: plt.Axes
    fig, ax = plt.subplots(layout="constrained")
    ax.boxplot(b_limsups, vert=False, patch_artist=True, labels=labels)
    fig.savefig(plot_dir / "limsup__pol_dt={}_c{}k.pdf".format(pol_dt, ckpt_k))
    plt.close(fig)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
