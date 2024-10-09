import functools as ft
import pathlib

import ipdb
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import typer
from loguru import logger

import run_config.int_avoid.segway_cfg
import run_config.nclf.pend_cfg
import run_config.nclf_pol.pend_cfg
import wandb
from clfrl.dyn.segway import Segway
from clfrl.ncbf.compute_disc_avoid import compute_all_disc_avoid_terms, cum_max_h
from clfrl.ncbf.int_avoid import IntAvoid
from clfrl.nclf.min_norm_control import bangbang_control
from clfrl.nclf.sim_nclf import SimNCLF
from clfrl.plotting.legend_helpers import lline
from clfrl.plotting.plot_task_summary import plot_task_summary
from clfrl.plotting.plot_utils import plot_boundaries
from clfrl.plotting.plotter import Plotter
from clfrl.solvers.qp import OSQPStatus
from clfrl.training.ckpt_manager import get_ckpt_manager, save_create_args
from clfrl.training.run_dir import init_wandb_and_get_run_dir
from clfrl.utils.ckpt_utils import get_id_from_ckpt, get_run_path_from_ckpt, load_ckpt
from clfrl.utils.jax_utils import jax2np, jax_jit, jax_vmap, rep_vmap
from clfrl.utils.logging import set_logger_format


def main(ckpt_path: pathlib.Path):
    set_logger_format()
    seed = 0

    run_path = get_run_path_from_ckpt(ckpt_path)
    plot_dir = run_path / "plots"

    # T = 600
    T = 200
    # T = 50

    task = Segway()
    CFG = run_config.int_avoid.segway_cfg.get(seed)
    nom_pol = task.nom_pol_lqr

    alg: IntAvoid = IntAvoid.create(seed, task, CFG.alg_cfg, nom_pol)
    alg = load_ckpt(alg, ckpt_path)
    logger.info("Loaded ckpt from {}!".format(ckpt_path))

    cid = get_id_from_ckpt(ckpt_path)

    # Try and see whether the MSE is killing us due to the long horizon?
    # x0 = np.array([0.0, -1.0, 0.0, -3.0])
    x0 = np.array([0.0, 0.5, 0.0, -3.4])
    logger.info("V_nom")
    bb_V_nom = jax2np(alg.get_bb_V_nom(T))

    bb_x, bb_Xs, bb_Ys = task.get_contour_x0()

    logger.info("Rollout cbf")
    alpha_safe, alpha_unsafe = 0.1, 100.0
    sim = SimNCLF(task, ft.partial(alg.get_cbf_control_sloped, alpha_safe, alpha_unsafe), T)
    T_x, T_u, T_t = jax2np(jax_jit(sim.rollout_plot)(x0))

    logger.info("Rollout nom")
    sim = SimNCLF(task, nom_pol, T)
    T_x_nom, T_u_nom, T_t_nom = jax2np(jax_jit(sim.rollout_plot)(x0))

    dt = np.diff(T_t).mean()
    n_skip = int(round(task.dt / dt))

    def get_info(state):
        h_V = alg.get_Vh(state)
        hx_Vx = jax.jacobian(alg.get_Vh)(state)
        assert hx_Vx.shape == (task.nh, task.nx)
        h_h = task.h_components(state)

        f, G = alg.task.f(state), alg.task.G(state)
        u_nom = alg.nom_pol(state)
        xdot_nom = f + jnp.sum(G * u_nom, axis=-1)

        h_Vdot_nom = jnp.sum(hx_Vx * xdot_nom, axis=-1)
        h_Vdot_disc_nom = h_Vdot_nom - alg.lam * (h_V - h_h)

        h_constr_nom = h_Vdot_nom + alpha_safe * h_V

        return h_h, h_V, h_Vdot_nom, h_Vdot_disc_nom, h_constr_nom

    def debug_qp(state):
        # h_V = alg.get_Vh(state)
        # hx_Vx = jax.jacobian(alg.get_Vh)(state)
        # assert hx_Vx.shape == (task.nh, task.nx)
        # h_h = task.h_components(state)
        # f, G = alg.task.f(state), alg.task.G(state)
        # u_nom = alg.nom_pol(state)
        u, (r, info) = alg.get_cbf_control_sloped_all(alpha_safe, alpha_unsafe, state)

        return r

    logger.info("get_info...")
    Th_h, Th_V, _, _, _ = jax_jit(jax_vmap(get_info))(T_x)
    Th_h_nom, Th_V_nom, Th_Vdot_nom, Th_Vdot_disc_nom, Th_constr_nom = jax_jit(jax_vmap(get_info))(T_x_nom)

    Th_V_true = jax2np(cum_max_h(Th_h_nom))
    Th_V_disc_true, _, _ = jax2np(compute_all_disc_avoid_terms(alg.lam, task.dt, Th_h_nom))

    logger.info("Getting qp resids...")
    T_r = jax2np(jax_vmap(debug_qp)(T_x))

    #####################################################
    logger.info("Plotting...")
    x_labels, h_labels = task.x_labels, task.h_labels

    traj_dt = np.diff(T_t_nom).mean()
    markevery = int(task.dt / traj_dt)
    logger.info("markevery = {}".format(markevery))
    mark_cfg = dict(marker="o", ms=1.5, markevery=markevery)

    # Show the phase.
    fig, ax = plt.subplots(layout="constrained")
    ax.plot(T_x_nom[:, task.TH], T_x_nom[:, task.W], color="C1", alpha=0.5, **mark_cfg)
    ax.plot(T_x[:, task.TH], T_x[:, task.W], color="C0", alpha=0.5, **mark_cfg)
    task.plot_pend(ax)
    ax.contour(bb_Xs, bb_Ys, bb_V_nom, levels=[0], colors=["C5"], zorder=3.5)
    fig.savefig(plot_dir / f"dbg_phase{cid}.pdf")
    plt.close(fig)

    # Plot trajectory. Also plot the train sample boundary.
    fig, axes = plt.subplots(task.nx, layout="constrained")
    for ii, ax in enumerate(axes):
        ax.plot(T_t_nom, T_x_nom[:, ii], color="C1", alpha=0.5, **mark_cfg)
        ylim = ax.get_ylim()
        ax.plot(T_t, T_x[:, ii], color="C0", alpha=0.5, **mark_cfg)
        ax.set(ylim=ylim)
        ax.set(title=x_labels[ii])
    plot_boundaries(axes, task.train_bounds())

    lines = [lline("C1")]
    axes[-1].legend(lines, ["Nom"], loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(plot_dir / f"eval_traj{cid}.pdf")
    plt.close(fig)

    # Plot V.
    fig, axes = plt.subplots(task.nh, layout="constrained")
    for ii, ax in enumerate(axes):
        ax.plot(T_t_nom[:], Th_V_nom[:, ii], color="C1", alpha=0.5, zorder=5, **mark_cfg)
        ax.plot(T_t_nom, Th_V_true[:, ii], color="C4", ls="--")
        ax.plot(T_t_nom, Th_V_disc_true[:, ii], color="C6", ls="--")
        ax.plot(T_t_nom, Th_h_nom[:, ii], color="C5", ls="-.")
        ylim = ax.get_ylim()
        ax.plot(T_t[:], Th_V[:, ii], color="C0", alpha=0.5, zorder=5, **mark_cfg)
        ax.set(ylim=ylim)

        ax.set(title=h_labels[ii])
    lines = [lline("C0"), lline("C1"), lline("C4"), lline("C6"), lline("C5")]
    axes[-1].legend(
        lines, ["QP", "Nom", "V True", "V disc true" "h"], loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.0)
    )
    fig.savefig(plot_dir / f"eval_traj_V{cid}.pdf")
    plt.close(fig)

    # Plot Vdot.
    fig, axes = plt.subplots(task.nh, layout="constrained")
    for ii, ax in enumerate(axes):
        ax.plot(T_t_nom[:], Th_Vdot_nom[:, ii], color="C1", alpha=0.5, zorder=5, **mark_cfg)
        # ax.plot(T_t_nom[:], Th_Vdot[:, ii], color="C0", alpha=0.5, zorder=5, **mark_cfg)
        ax.set(title=h_labels[ii])
    lines = [lline("C0"), lline("C1")]
    axes[-1].legend(lines, ["QP", "Nom"], loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(plot_dir / f"eval_traj_Vdot{cid}.pdf")
    plt.close(fig)

    # Plot constraints.
    fig, axes = plt.subplots(task.nh + task.nu + 1, layout="constrained")
    for ii, ax in enumerate(axes[: task.nh]):
        ax.plot(T_t_nom[:], Th_constr_nom[:, ii], color="C1", alpha=0.5, zorder=5, **mark_cfg)
        ax.set(title=h_labels[ii])

    axes[-2].plot(T_t_nom[:-1:n_skip], T_u_nom[:, 0], color="C1", alpha=0.5, zorder=5, **mark_cfg)
    axes[-2].plot(T_t[:-1:n_skip], T_u[:, 0], color="C0", alpha=0.5, zorder=5, **mark_cfg)
    lines = [lline("C0"), lline("C1")]
    axes[-2].legend(lines, ["QP", "Nom"], loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.0))

    axes[-1].plot(T_t, T_r, color="C0", alpha=0.5, zorder=5, **mark_cfg)
    axes[-1].set_ylim(-2e-1, 2e-1)

    fig.savefig(plot_dir / f"eval_traj_constr{cid}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)
