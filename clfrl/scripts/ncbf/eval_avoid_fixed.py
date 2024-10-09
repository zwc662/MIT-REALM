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
import run_config.nclf.pend_cfg
import run_config.nclf_pol.pend_cfg
from clfrl.dyn.doubleint_wall import DoubleIntWall
from clfrl.ncbf.avoid_fixed import AvoidFixed
from clfrl.nclf.min_norm_control import bangbang_control
from clfrl.nclf.sim_nclf import SimNCLF
from clfrl.solvers.qp import OSQPStatus
from clfrl.utils.ckpt_utils import get_run_path_from_ckpt, load_ckpt
from clfrl.utils.jax_utils import jax2np, jax_jit, jax_vmap
from clfrl.utils.logging import set_logger_format


def main(ckpt_path: pathlib.Path):
    set_logger_format()
    seed = 0

    task = DoubleIntWall()

    # nom_pol = task.nom_pol_rng
    nom_pol = task.nom_pol_rng2
    # nom_pol = task.nom_pol_osc

    CFG = run_config.avoid_fixed.doubleintwall_cfg.get(seed)
    alg: AvoidFixed = AvoidFixed.create(seed, task, CFG.alg_cfg, nom_pol)
    alg = load_ckpt(alg, ckpt_path)
    logger.info("Loaded ckpt from {}!".format(ckpt_path))

    # x0 = np.array([-0.8, 1.5])
    x0 = np.array([-0.6, 1.7])
    T = 80

    # Rollout in the way it's done in eval().
    # alpha = 0.1
    alpha = 1.0

    # alpha_safe, alpha_unsafe = alpha, alpha
    alpha_safe, alpha_unsafe = 0.1, 100.0

    # sim = SimNCLF(task, ft.partial(alg.get_cbf_control, alpha), T)
    sim = SimNCLF(task, ft.partial(alg.get_cbf_control_sloped, alpha_safe, alpha_unsafe), T)
    T_x, T_u, T_t = jax2np(jax_jit(sim.rollout_plot)(x0))

    # Also visualize the original nominal policy.
    sim = SimNCLF(task, nom_pol, T)
    T_x_nom, _, T_t_nom = jax2np(jax_jit(sim.rollout_plot)(x0))

    # Also visualize the bang-bang policy.
    sim = SimNCLF(task, alg.get_opt_u, T)
    T_x_opt, _, T_t_nom = jax2np(jax_jit(sim.rollout_plot)(x0))

    _, bb_Xs, bb_Ys = task.get_contour_x0()
    bb_V_nom = jax2np(jax_jit(alg.get_bb_V_nom)())
    bb_V_opt = jax2np(jax_jit(alg.get_bb_V_opt)())

    def get_info(state):
        get_Vh = ft.partial(alg.get_Vh, alg.V.params)
        get_V = ft.partial(alg.get_V, alg.V.params)
        h_V = get_Vh(state)
        hx_Vx = jax.jacobian(get_Vh)(state)
        assert hx_Vx.shape == (task.nh, task.nx)

        V = get_V(state)
        Vx = jax.grad(get_V)(state)

        f, G = alg.task.f(state), alg.task.G(state)
        # u, (r, (qpstate, (_, _, qpG, qph))) = alg.get_cbf_control_all(alpha, state)
        u, (r, (qpstate, (_, _, qpG, qph))) = alg.get_cbf_control_sloped_all(alpha_safe, alpha_unsafe, state)
        xdot = f + jnp.sum(G * u, axis=-1)
        u_nom = alg.nom_pol(state)
        xdot_nom = f + jnp.sum(G * u_nom, axis=-1)

        qp_x = jnp.array([*u, r])
        lhs = qpG @ qp_x
        rhs = qph

        task.chk_x(xdot)

        Vdot = jnp.dot(Vx, xdot)
        constr = Vdot - alpha * V

        h_Vdot = jnp.sum(hx_Vx * xdot, axis=-1)
        h_Vdot_nom = jnp.sum(hx_Vx * xdot_nom, axis=-1)

        u_lb, u_ub = np.array([-1.0]), np.array([+1.0])
        u_bb = bangbang_control(u_lb, u_ub, Vx, G)
        xdot_bb = f + jnp.sum(G * u_bb, axis=-1)
        h_Vdot_min = jnp.sum(hx_Vx * xdot_bb, axis=-1)

        # ipdb.set_trace()
        return h_V, hx_Vx, h_Vdot, h_Vdot_nom, h_Vdot_min, constr, u, r, (lhs, rhs), qpstate

    # get_info(T_x[0])
    Th_V, Thx_Vx, Th_Vdot, Th_Vdot_nom, Th_Vdot_min, T_constr, T_u, T_r, (T_lhs, T_rhs), T_qpstate = jax_jit(
        jax_vmap(get_info)
    )(T_x)

    # def get_qp_mats(state):
    #     u, (r, (qpstate, (qpQ, qpc, qpG, qph))) = alg.get_cbf_control_sloped_all(alpha_safe, alpha_unsafe, state)
    #     sol = jnp.concatenate([u, jnp.array([r])])
    #     return qpstate, sol, (qpQ, qpc, qpG, qph)
    #
    # bad_idxs = T_qpstate.status != OSQPStatus.SOLVED
    # b_qpstate, b_sol, (b_qpQ, b_qpc, b_qpG, b_qph) = jax_jit(jax_vmap(get_qp_mats))(T_x[bad_idxs])
    #
    # np.savez("bad_qps.npz", Q=b_qpQ, c=b_qpc, G=b_qpG, h=b_qph, sol=b_sol)
    # print("Saved to bad_qps.npz!")
    # return

    #####################################################
    run_path = get_run_path_from_ckpt(ckpt_path)
    plot_dir = run_path / "plots"
    #####################################################
    # Plot phase.

    fig, ax = plt.subplots(layout="constrained")
    ax.plot(T_x[:, 0], T_x[:, 1], color="C1", marker="o", ms=0.8, lw=0.5, zorder=4, label="QP")
    ax.scatter(T_x[0, 0], T_x[0, 1], color="black", s=1**2, zorder=6, marker="s")
    ax.plot(T_x_opt[:, 0], T_x_opt[:, 1], color="C4", marker="o", ms=0.8, lw=0.5, zorder=4, label="Opt")
    ax.plot(T_x_nom[:, 0], T_x_nom[:, 1], color="C3", marker="o", ms=0.8, lw=0.5, zorder=4, label="Nominal")

    ax.contour(bb_Xs, bb_Ys, bb_V_nom, levels=[0], colors=["C5"], zorder=3.5, label="Nominal CI")
    ax.contour(bb_Xs, bb_Ys, bb_V_opt, levels=[0], colors=["C6"], zorder=3.5, label="Opt CI")

    task.plot_phase(ax)
    fig.savefig(plot_dir / "eval_phase.pdf")
    fig.savefig(plot_dir / "eval_phase.png")
    plt.close(fig)
    #####################################################
    # Plot how V evolves through the traj.
    figsize = np.array([4.0, 8.0])
    fig, axes = plt.subplots(7, figsize=figsize, layout="constrained")
    axes[0].plot(T_t, Th_V[:, 0], color="C1", lw=0.5, marker="o", ms=1, ls="--", alpha=0.8)
    axes[0].set(title=r"$V[0]$")
    axes[1].plot(T_t, Th_V[:, 1], color="C1", lw=0.5, marker="o", ms=1, ls="--", alpha=0.8)
    axes[1].set(title=r"$V[1]$")

    axes[2].plot(T_t, Th_Vdot[:, 0], color="C1", lw=0.5, ls="--", alpha=0.8)
    axes[2].plot(T_t, Th_Vdot_nom[:, 0], color="C0", lw=0.5, ls="--", alpha=0.8)
    axes[2].plot(T_t, -alpha * Th_V[:, 0], color="C3", lw=0.5, ls="--", alpha=0.5)
    axes[2].plot(T_t, Th_Vdot_min[:, 0], color="C5", lw=0.5, ls="--", alpha=0.8)
    axes[2].set(title=r"$\dot{V}[0]$")

    axes[3].plot(T_t, Th_Vdot[:, 1], color="C1", lw=0.5, ls="--", alpha=0.8)
    axes[3].plot(T_t, Th_Vdot_nom[:, 1], color="C0", lw=0.5, ls="--", alpha=0.8)
    axes[3].plot(T_t, -alpha * Th_V[:, 1], color="C3", lw=0.5, ls="--", alpha=0.8)
    axes[3].plot(T_t, Th_Vdot_min[:, 1], color="C5", lw=0.5, ls="--", alpha=0.8)
    axes[3].set(title=r"$\dot{V}[1]$")

    axes[4].plot(T_t, T_constr, color="C0", lw=0.5, marker="o", ms=1, ls="--", alpha=0.8, label="orig")
    axes[4].plot(T_t, T_constr - T_r, color="C1", lw=0.5, marker="o", ms=1, ls="--", alpha=0.8, label="relaxed")
    axes[4].set(title="Constraint")
    axes[4].legend()

    Tb_constr_resid = T_lhs - T_rhs
    n_constrs = Tb_constr_resid.shape[1]
    for ii in range(2):
        axes[5].plot(T_t, Tb_constr_resid[:, ii], lw=0.5, marker="o", ms=1, ls="--", alpha=0.8, label=f"{ii}")
    axes[5].legend()
    axes[5].set(title="Constraint")

    axes[6].plot(T_t, T_r, color="C1", lw=0.5, marker="o", ms=1, ls="--", alpha=0.8, label="relaxed")
    axes[6].set(ylim=(-0.1, 0.1))
    axes[6].set(title="QP relax")

    axes[-1].set(xlabel="Time (s)")
    fig.savefig(plot_dir / "eval_V.pdf")

    print(T_qpstate)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)
