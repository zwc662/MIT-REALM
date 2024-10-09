import functools as ft
import pathlib

import ipdb
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import typer
from loguru import logger

import run_config.avoid_fixed.doubleintwall_cfg
import run_config.avoid_fixed.segway_cfg
import run_config.nclf.pend_cfg
import run_config.nclf_pol.pend_cfg
from clfrl.dyn.segway import Segway
from clfrl.ncbf.avoid_fixed import AvoidFixed
from clfrl.ncbf.compute_disc_avoid import cum_max_h
from clfrl.nclf.sim_nclf import SimNCLF
from clfrl.plotting.legend_helpers import lline
from clfrl.utils.ckpt_utils import get_run_path_from_ckpt, load_ckpt
from clfrl.utils.jax_utils import jax2np, jax_jit, jax_vmap
from clfrl.utils.logging import set_logger_format


def main(ckpt_path: pathlib.Path):
    set_logger_format()
    seed = 0

    run_path = get_run_path_from_ckpt(ckpt_path)
    plot_dir = run_path / "plots"

    task = Segway()
    # task._dt_orig /= 12

    CFG = run_config.avoid_fixed.segway_cfg.get(seed)
    CFG.alg_cfg.eval_cfg.eval_rollout_T = 600
    nom_pol = task.nom_pol_lqr

    alg: AvoidFixed = AvoidFixed.create(seed, task, CFG.alg_cfg, nom_pol)
    alg = load_ckpt(alg, ckpt_path)
    logger.info("Loaded ckpt from {}!".format(ckpt_path))

    T = 600
    # T = 60
    # x0 = np.array([0.0, 0.0, 0.0, -2.5])
    # x0 = np.array([0.0, 0.0, 0.0, -2.5])
    x0 = np.array([0.0, -0.5, 0.0, 0.0])

    alpha_safe, alpha_unsafe = 0.1, 100.0

    logger.info("Rollout cbf")
    sim = SimNCLF(task, ft.partial(alg.get_cbf_control_sloped, alpha_safe, alpha_unsafe), T)
    T_x, T_u, T_t = jax2np(jax_jit(sim.rollout_plot)(x0))

    logger.info("Rollout nom")
    sim = SimNCLF(task, nom_pol, T)
    T_x_nom, _, T_t_nom = jax2np(jax_jit(sim.rollout_plot)(x0))

    def get_info(state):
        get_Vh = ft.partial(alg.get_Vh, alg.V.params)
        h_V = get_Vh(state)
        hx_Vx = jax.jacobian(get_Vh)(state)
        assert hx_Vx.shape == (task.nh, task.nx)

        f, G = alg.task.f(state), alg.task.G(state)
        u, (r, (qpstate, (_, _, qpG, qph))) = alg.get_cbf_control_sloped_all(alpha_safe, alpha_unsafe, state)
        xdot = f + jnp.sum(G * u, axis=-1)
        u_nom = alg.nom_pol(state)
        xdot_nom = f + jnp.sum(G * u_nom, axis=-1)

        h_Vdot = jnp.sum(hx_Vx * xdot, axis=-1)
        h_Vdot_nom = jnp.sum(hx_Vx * xdot_nom, axis=-1)

        h_h = task.h_components(state)

        return h_h, h_V, h_Vdot, h_Vdot_nom, u, r

    logger.info("get_info...")
    Th_h, Th_V, Th_Vdot, Th_Vdot_nom, T_u, T_r = jax_jit(jax_vmap(get_info))(T_x)
    Th_h_nom, Th_V_nom, _, _, _, _ = jax_jit(jax_vmap(get_info))(T_x_nom)

    Th_V_true = jax2np(cum_max_h(Th_h_nom))
    # wtf = cum_max(Th_h_nom[:, 0])
    # fig, ax = plt.subplots()
    # ax.plot(wtf, alpha=0.2, label="wtf")
    # ax.plot(Th_h_nom[:, 0], ls="--", label="Th_h_now")
    # ax.plot(Th_V_true[:, 0], ls="-.", label="Th_V_true")
    # ax.legend()
    # fig.savefig("wtf.pdf")
    # ipdb.set_trace()

    #####################################################
    logger.info("Plotting...")
    x_labels, h_labels = task.x_labels, task.h_labels

    small_t = 3.0
    plot_T = np.argmin(T_t < small_t)

    traj_dt = np.diff(T_t).mean()
    markevery = int(task.dt / traj_dt)
    logger.info("markevery = {}".format(markevery))
    mark_cfg = dict(marker="o", ms=1.5, markevery=markevery)

    # Plot phase.
    fig, ax = plt.subplots(layout="constrained")
    ax.plot(T_x[:, task.TH], T_x[:, task.W], color="C1", marker="o", ms=0.8, lw=0.5, zorder=4)
    ax.plot(T_x_nom[:, task.TH], T_x_nom[:, task.W], color="C3", marker="o", ms=0.8, lw=0.5, zorder=4)
    lines = [lline("C1"), lline("C3")]
    ax.legend(lines, ["QP", "Nominal"], loc="upper right")
    fig.savefig(plot_dir / "eval_phase_dbg.pdf")
    plt.close(fig)

    # Plot trajectory.
    fig, axes = plt.subplots(task.nx, layout="constrained")
    for ii, ax in enumerate(axes):
        ax.plot(T_t[:plot_T], T_x[:plot_T, ii], color="C0", alpha=0.5, **mark_cfg)
        ax.plot(T_t_nom[:plot_T], T_x_nom[:plot_T, ii], color="C1", alpha=0.5, **mark_cfg)
        ax.set(title=x_labels[ii])
    lines = [lline("C0"), lline("C1")]
    axes[-1].legend(lines, ["QP", "Nom"], loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(plot_dir / "eval_traj.pdf")
    plt.close(fig)

    # Plot V.
    ax: plt.Axes
    fig, axes = plt.subplots(task.nh, layout="constrained")
    for ii, ax in enumerate(axes):
        ax.plot(T_t[:], Th_V[:, ii], color="C0", alpha=0.5, zorder=5, **mark_cfg)
        # ax.plot(T_t, Th_h[:, ii], color="C0", ls="--", lw=0.5, **mark_cfg)
        ax.plot(T_t_nom[:], Th_V_nom[:, ii], color="C1", alpha=0.5, zorder=5, **mark_cfg)
        # ax.plot(T_t_nom, Th_h_nom[:, ii], color="C1", ls="--", lw=0.5, **mark_cfg)
        ax.plot(T_t_nom, Th_V_true[:, ii], color="C4", ls="--")
        ax.plot(T_t_nom, Th_h_nom[:, ii], color="C5", ls="-.")
        ax.set(title=h_labels[ii])
    lines = [lline("C0"), lline("C1"), lline("C4"), lline("C5")]
    axes[-1].legend(lines, ["QP", "Nom", "V True", "h"], loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(plot_dir / "eval_traj_V.pdf")
    plt.close(fig)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)
