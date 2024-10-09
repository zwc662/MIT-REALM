import functools as ft
import pathlib

import ipdb
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import typer
from loguru import logger
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

import run_config.nclf.pend_cfg
import run_config.nclf_pol.pend_cfg
import wandb
from clfrl.dyn.pend import Pend
from clfrl.dyn.sim_cts import SimCts
from clfrl.nclf.nclf import NCLF
from clfrl.nclf.nclf_pol import NCLFPol
from clfrl.nclf.sim_nclf import SimNCLF
from clfrl.plotting.plot_task_summary import plot_task_summary
from clfrl.plotting.plotter import Plotter
from clfrl.training.ckpt_manager import get_ckpt_manager, save_create_args
from clfrl.training.run_dir import init_wandb_and_get_run_dir
from clfrl.utils.ckpt_utils import get_run_path_from_ckpt, load_ckpt
from clfrl.utils.jax_utils import jax2np, jax_jit, jax_vmap, rep_vmap
from clfrl.utils.logging import set_logger_format
from clfrl.utils.shape_utils import assert_shape


def main(ckpt_path: pathlib.Path):
    set_logger_format()
    seed = 0

    task = Pend()
    CFG = run_config.nclf_pol.pend_cfg.get(seed)
    CFG.alg_cfg.eval_cfg.eval_rollout_T = 128

    alg: NCLFPol = NCLFPol.create(seed, task, CFG.alg_cfg)
    alg = load_ckpt(alg, ckpt_path)
    logger.info("Loading ckpt from {}... Done!".format(ckpt_path))

    x0 = np.array([2.0, 0.5])

    # Rollout in the way it's done in eval().
    pol_mode = ft.partial(alg.get_control, alg.pol.params)
    # T = alg.cfg.eval_cfg.eval_rollout_T
    T = 5

    sim = SimNCLF(task, pol_mode, T)
    T_x, _, T_t = jax_jit(sim.rollout_plot)(x0)

    interp_pts = 64
    sim = SimCts(task, alg.pol.apply, T, interp_pts, pol_dt=task.dt)
    T_x_den, _, T_t_den = jax_jit(sim.rollout_plot)(x0)

    xmin, ymin = T_x.min(axis=0)
    xmax, ymax = T_x.max(axis=0)
    xr, yr = xmax - xmin, ymax - ymin
    xb, yb = 0.1 * xr, 0.1 * yr
    b_xs = np.linspace(xmin - xb, xmax + xb, num=128)
    b_ys = np.linspace(ymin - yb, ymax + yb, num=128)
    bb_Xs, bb_Ys = np.meshgrid(b_xs, b_ys)
    bb_state = np.stack([bb_Xs, bb_Ys], axis=-1)
    bb_V = jax2np(jax_jit(rep_vmap(alg.V.apply, rep=2))(bb_state))

    def get_info(state):
        V, Vx = jax.value_and_grad(ft.partial(alg.get_V, alg.V.params))(state)
        f, G = alg.task.f(state), alg.task.G(state)
        u = alg.get_control(alg.pol.params, state)
        xdot = f + jnp.sum(G * u, axis=-1)

        task.chk_x(xdot)
        task.chk_x(Vx)

        Vdot = jnp.dot(Vx, xdot)
        return V, Vx, Vdot, xdot, u

    T_V, T_Vx, T_Vdot, T_dx, T_u = jax_jit(jax_vmap(get_info))(T_x)
    T_V_den, T_Vx_den, T_Vdot_den, T_dx_den, T_u_den = jax_jit(jax_vmap(get_info))(T_x_den)

    #####################################################
    end_style = dict(s=1**2, zorder=7, marker="o")
    #####################################################
    run_path = get_run_path_from_ckpt(ckpt_path)
    plot_dir = run_path / "plots"

    fig, ax = plt.subplots(layout="constrained")
    ax.plot(T_x[:, 0], T_x[:, 1], color="C1", marker="o", ms=2, lw=0.5, zorder=4)
    ax.scatter(T_x[-1, 0], T_x[-1, 1], color="C5", **end_style)
    ax.scatter(T_x[0, 0], T_x[0, 1], color="black", s=1**2, zorder=6, marker="s")
    ax.contourf(bb_Xs, bb_Ys, bb_V, levels=14, norm=Normalize(vmin=0.0), cmap="rocket_r", alpha=0.8)
    ax.autoscale_view()
    fig.savefig(plot_dir / "eval_phase.pdf")
    plt.close(fig)

    # Plot how V evolves through the traj.
    figsize = np.array([4.0, 8.0])
    fig, axes = plt.subplots(7, figsize=figsize, layout="constrained")
    axes[0].plot(T_t, T_V, color="C1", lw=0.5, marker="o", ms=1, ls="--", alpha=0.8)
    axes[0].plot(T_t_den, T_V_den, color="C0", lw=0.5, alpha=0.8)
    axes[0].set(title=r"$V$")

    axes[1].plot(T_t, T_Vx[:, 0], color="C1", lw=0.5, marker="o", ms=1, ls="--", alpha=0.8)
    axes[1].plot(T_t_den, T_Vx_den[:, 0], color="C0", lw=0.5, alpha=0.8)
    axes[1].set_title(r"$V_\theta$")

    axes[2].plot(T_t, T_Vx[:, 1], color="C1", lw=0.5, marker="o", ms=1, ls="--", alpha=0.8)
    axes[2].plot(T_t_den, T_Vx_den[:, 1], color="C0", lw=0.5, alpha=0.8)
    axes[2].set_title(r"$V_\omega$")

    axes[3].plot(T_t, T_dx[:, 0], color="C1", lw=0.5, marker="o", ms=1, ls="--", alpha=0.8)
    axes[3].plot(T_t_den, T_dx_den[:, 0], color="C0", lw=0.5, alpha=0.8)
    axes[3].set_title(r"$\dot{\theta}$")

    axes[4].plot(T_t, T_dx[:, 1], color="C1", lw=0.5, marker="o", ms=1, ls="--", alpha=0.8)
    axes[4].plot(T_t_den, T_dx_den[:, 1], color="C0", lw=0.5, alpha=0.8)
    axes[4].set_title(r"$\dot{\omega}$")

    axes[5].plot(T_t, T_u[:, 1], color="C1", lw=0.5, marker="o", ms=1, ls="--", alpha=0.8)
    axes[5].plot(T_t_den, T_u_den[:, 1], color="C0", lw=0.5, alpha=0.8)
    axes[5].set_title(r"$u$")

    axes[6].plot(T_t, T_Vdot, color="C1", lw=0.5, ls="--", alpha=0.8)
    axes[6].plot(T_t_den, T_Vdot_den, color="C0", lw=0.5, alpha=0.8)
    axes[6].set(title=r"$\dot{V}$")

    axes[-1].set(xlabel="Time (s)")

    for kk in range(T):
        [ax.axvline(kk * task.dt, color="C2", lw=0.8, alpha=0.5) for ax in axes]

    fig.savefig(plot_dir / "eval_V.pdf")


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)
