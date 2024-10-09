import functools as ft
import pathlib

import ipdb
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import typer
from loguru import logger
from matplotlib.colors import CenteredNorm

import run_config.int_avoid.doubleintwall_cfg
import run_config.nclf.pend_cfg
import run_config.nclf_pol.pend_cfg
from clfrl.dyn.doubleint_wall import DoubleIntWall
from clfrl.dyn.sim_cts import SimCtsReal
from clfrl.ncbf.int_avoid import IntAvoid
from clfrl.nclf.sim_nclf import SimNCLF
from clfrl.plotting.contour_utils import centered_norm
from clfrl.plotting.plotstyle import PlotStyle
from clfrl.utils.ckpt_utils import get_run_path_from_ckpt, load_ckpt
from clfrl.utils.jax_utils import jax2np, jax_jit, jax_vmap, rep_vmap
from clfrl.utils.logging import set_logger_format


def main(ckpt_path: pathlib.Path):
    set_logger_format()
    seed = 0

    run_path = get_run_path_from_ckpt(ckpt_path)
    plot_dir = run_path / "plots"

    task = DoubleIntWall()

    nom_pol = task.nom_pol_osc

    CFG = run_config.int_avoid.doubleintwall_cfg.get(seed)
    alg: IntAvoid = IntAvoid.create(seed, task, CFG.alg_cfg, nom_pol)
    alg = load_ckpt(alg, ckpt_path)
    logger.info("Loaded ckpt from {}!".format(ckpt_path))

    # Plot how V varies along a trajectory.
    # x0 = np.array([-0.6, 1.7])
    x0 = np.array([0.5, -1.7])
    T = 80
    tf = T * task.dt

    # Original nominal policy.
    logger.info("Sim nom...")
    sim = SimCtsReal(task, nom_pol, tf, task.dt, use_pid=True)
    T_x_nom, T_t_nom, _ = jax2np(jax_jit(sim.rollout_plot)(x0))

    alphas = np.array([0.1, 1.0, 5.0, 10.0])

    def int_pol_for_alpha(alpha_safe):
        alpha_unsafe = 10.0
        pol = ft.partial(alg.get_cbf_control_sloped, alpha_safe, alpha_unsafe)
        sim = SimCtsReal(task, pol, tf, 0.5 * task.dt, use_obs=False, use_pid=False, max_steps=512)
        T_x, T_t, _ = sim.rollout_plot(x0)
        return T_x, T_t

    logger.info("Sim pol for different alphas...")
    bT_x, bT_t = [], []
    for alpha in alphas:
        T_x, T_t = jax2np(jax_jit(int_pol_for_alpha)(alpha))
        bT_x.append(T_x)
        bT_t.append(T_t)
    bT_x = np.stack(bT_x, axis=0)
    bT_t = np.stack(bT_t, axis=0)

    logger.info("bbh_Vh...")
    bb_x, bb_Xs, bb_Ys = task.get_contour_x0(n_pts=192)
    bbh_Vh = jax2np(jax_jit(rep_vmap(alg.get_Vh, rep=2))(bb_x))
    bb_Vh = bbh_Vh.max(-1)

    def get_info(state):
        h_V = alg.get_Vh(state)
        hx_Vx = jax.jacobian(alg.get_Vh)(state)
        assert hx_Vx.shape == (task.nh, task.nx)
        h_h = alg.task.h_components(state)

        f, G = alg.task.f(state), alg.task.G(state)
        u_nom = alg.nom_pol(state)

        xdot_nom = f + jnp.sum(G * u_nom, axis=-1)
        h_Vdot_nom = jnp.sum(hx_Vx * xdot_nom, axis=-1)
        h_Vdot_nom_disc = h_Vdot_nom - alg.lam * (h_V - h_h)

        return h_V, h_Vdot_nom, h_Vdot_nom_disc

    logger.info("get_info...")
    Th_V, Th_Vdot_nom, Th_Vdot_nom_disc = jax2np(jax_jit(jax_vmap(get_info))(T_x_nom))

    # Plot L_G V.
    def get_Lg_V(state):
        # (nh, nx)
        hx_Vx = jax.jacobian(alg.get_Vh)(state)
        # (nx, nu)
        G = alg.task.G(state)
        # (nh, nu) -> (nh,)
        h_LG_V = (hx_Vx @ G).flatten()
        return h_LG_V

    bbh_LGV = jax2np(rep_vmap(get_Lg_V, rep=2)(bb_x))

    #####################################################
    logger.info("Plotting...")
    h_labels = task.h_labels

    fig, ax = plt.subplots(layout="constrained")
    ax.plot(T_x_nom[:, 0], T_x_nom[:, 1], color="C3", ls="--", label="Nominal")
    for ii, alpha in enumerate(alphas):
        ax.plot(bT_x[ii, :, 0], bT_x[ii, :, 1], color=f"C{ii}", label=f"QP ({alpha})")
    ax.contour(bb_Xs, bb_Ys, bb_Vh, levels=[0.0], colors=[PlotStyle.ZeroColor], alpha=0.6, linewidths=1.0)
    task.plot_phase(ax)
    ax.legend()
    fig.savefig(plot_dir / "eval_phase.pdf")
    plt.close(fig)

    norm = centered_norm(bbh_Vh.min(), bbh_Vh.max())

    levels = 31
    figsize = (8, 4)
    fig, axes = plt.subplots(1, task.nh, figsize=figsize, layout="constrained")
    for ii, ax in enumerate(axes):
        cs0 = ax.contourf(bb_Xs, bb_Ys, bbh_Vh[:, :, ii], norm=norm, levels=levels, cmap="RdBu_r", alpha=0.9)
        cs1 = ax.contour(
            bb_Xs, bb_Ys, bbh_Vh[:, :, ii], levels=[0.0], colors=[PlotStyle.ZeroColor], alpha=0.98, linewidths=1.0
        )
        cbar = fig.colorbar(cs0, ax=ax)
        cbar.add_lines(cs1)
        task.plot_phase(ax)
        ax.set_title(h_labels[ii])
    fig.savefig(plot_dir / "eval_Vh.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(task.nh, layout="constrained")
    for ii, ax in enumerate(axes):
        ax.plot(T_t_nom, Th_V[:, ii])
        ax.set_title(h_labels[ii])
    fig.savefig(plot_dir / "eval_V_traj.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(task.nh, layout="constrained")
    for ii, ax in enumerate(axes):
        ax.plot(T_t_nom, Th_Vdot_nom_disc[:, ii])
        ax.set_title(h_labels[ii])
    fig.savefig(plot_dir / "eval_Vdot_nom_disc_traj.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, task.nh, layout="constrained")
    for ii, ax in enumerate(axes):
        ax.contourf(bb_Xs, bb_Ys, bbh_LGV[:, :, ii], levels=9, norm=CenteredNorm(), cmap="RdBu_r", alpha=0.9)
        task.plot_phase(ax)
        ax.set_title(h_labels[ii])
    fig.suptitle("LG V")
    fig.savefig(plot_dir / "LG_V.pdf")
    plt.close(fig)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)
