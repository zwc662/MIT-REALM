import functools as ft
import pathlib
import pickle

import ipdb
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import typer
from jax_f16.f16 import F16
from loguru import logger
from matplotlib.colors import CenteredNorm

import run_config.int_avoid.f16two_cfg
import run_config.nclf.pend_cfg
import run_config.nclf_pol.pend_cfg
from pncbf.dyn.dyn_types import Control, State
from pncbf.dyn.f16_two import A_BOUNDS, B_BOUNDS, F16Two
from pncbf.dyn.sim_cts_pbar import SimCtsPbar
from pncbf.ncbf.compute_disc_avoid import AllDiscAvoidTerms, compute_all_disc_avoid_terms
from pncbf.ncbf.int_avoid import IntAvoid
from pncbf.plotting.legend_helpers import lline
from pncbf.plotting.plot_utils import plot_boundaries
from pncbf.plotting.plotstyle import PlotStyle
from pncbf.utils.angle_utils import wrap_to_pi
from pncbf.utils.ckpt_utils import get_id_from_ckpt, get_run_path_from_ckpt, load_ckpt_with_step
from pncbf.utils.easy_npz import EasyNpz, save_data
from pncbf.utils.jax_utils import jax2np, jax_default_x32, jax_jit_np, jax_vmap, rep_vmap
from pncbf.utils.logging import set_logger_format
from pncbf.utils.path_utils import mkdir

app = typer.Typer()


@app.command()
def gen(ckpt_path: pathlib.Path):
    jax_default_x32()
    set_logger_format()
    task = F16Two()

    run_path = get_run_path_from_ckpt(ckpt_path)
    plot_dir = mkdir(run_path / "eval/dbg_qp")

    CFG = run_config.int_avoid.f16two_cfg.get(0)
    nom_pol = task.nom_pol_N0_pid
    alg: IntAvoid = IntAvoid.create(0, task, CFG.alg_cfg, nom_pol)
    alg, ckpt_path = load_ckpt_with_step(alg, ckpt_path)
    logger.info("Loaded ckpt from {}!".format(ckpt_path))
    cid = get_id_from_ckpt(ckpt_path)

    tf = 16.0
    dt = task.dt / 4
    n_steps = int(round(tf / dt))

    x0 = task.nominal_val_state()
    # x0[task.PE0] = -1_500
    # x0[task.PE0] = -2000
    # x0[task.PE0] = -2_300
    # x0[task.H0] = 350.0
    # x0[task.PN0] += 50

    x0[task.PE0] = -2_500
    x0[task.PN0] = -200
    # x0[task.PN0] = -400
    # x0[task.PSI0] = 0.75 * np.pi

    x0[task.PE1] = 0.0
    x0[task.PN1] = 1_000.0
    x0[task.PSI1] = -0.75 * np.pi

    # pol = task.nom_pol_pid

    sim = SimCtsPbar(task, task.nom_pol_pid, n_steps, dt, dt0=dt, use_obs=False, use_pid=False, max_steps=n_steps, solver="bosh3")
    T_x_nom, T_t_nom = jax_jit_np(sim.rollout_plot)(x0)

    Th_h_nom = jax2np(jax_vmap(task.h_components)(T_x_nom))
    Th_Vh_pred_nom = jax_jit_np(jax_vmap(alg.get_Vh))(T_x_nom)
    Teh_Vh_tgt_nom = jax_jit_np(jax_vmap(alg.get_eVh_tgt))(T_x_nom)

    #####################################################################
    test_pol = task.nom_pol_N0_pid

    V_shift = -(1 - np.exp(-alg.lam * task.max_ttc)) * task.h_min
    logger.info("lam: {}, V_shift: {}".format(alg.lam, V_shift))

    alpha_safe = np.full(task.nh, 2.0)
    alpha_unsafe = np.full(task.nh, 100.0)
    pol = ft.partial(alg.get_cbf_control_sloped, alpha_safe, alpha_unsafe, nom_pol=test_pol, V_shift=V_shift)
    sim = SimCtsPbar(task, pol, n_steps, dt, dt0=dt, use_obs=False, use_pid=False, max_steps=n_steps, solver="bosh3")
    T_x, T_t = jax_jit_np(sim.rollout_plot)(x0)

    Th_h = jax2np(jax_vmap(task.h_components)(T_x))
    Th_Vh_pred = jax_jit_np(jax_vmap(alg.get_Vh))(T_x)
    Teh_Vh_tgt = jax_jit_np(jax_vmap(alg.get_eVh_tgt))(T_x)

    # For better visualization purposes, compute the nominal Vh.
    bb_V_noms = jax2np(task.get_bb_V_noms(n_steps, nom_pol))

    # Save results.
    pkl_path = plot_dir / f"data{cid}.pkl"
    data = dict(
        T_t=T_t,
        T_x=T_x,
        T_x_nom=T_x_nom,
        Th_h=Th_h,
        Th_h_nom=Th_h_nom,
        Th_Vh_pred=Th_Vh_pred,
        Th_Vh_pred_nom=Th_Vh_pred_nom,
        Teh_Vh_tgt=Teh_Vh_tgt,
        bb_V_noms=bb_V_noms,
    )
    save_data(pkl_path, **data)

    plot(ckpt_path)


@app.command()
def plot(ckpt_path: pathlib.Path):
    jax_default_x32()
    set_logger_format()
    task = F16Two()

    cid = get_id_from_ckpt(ckpt_path)

    run_path = get_run_path_from_ckpt(ckpt_path)
    plot_dir = mkdir(run_path / "eval/dbg_qp")
    npz_path = plot_dir / f"data{cid}.pkl"
    npz = EasyNpz(npz_path)

    T_t, T_x, T_x_nom = npz("T_t", "T_x", "T_x_nom")
    Th_Vh_pred, Th_Vh_pred_nom = npz("Th_Vh_pred", "Th_Vh_pred_nom")
    bb_V_noms = npz("bb_V_noms")

    Th_h, Th_h_nom = npz("Th_h", "Th_h_nom")

    T_Vh_pred = np.max(Th_Vh_pred, axis=1)
    T_Vh_pred_nom = np.max(Th_Vh_pred_nom, axis=1)

    # Store the first index where we go out of the training bounds.
    train_bounds = task.train_bounds()
    Tx_oob = (T_x < train_bounds[0]) | (T_x > train_bounds[1])
    # positions and heading angle do not count.
    Tx_oob[:, [task.PN0, task.PE0, task.PSI0]] = False
    if train_bounds[1, task.PHI0] - train_bounds[0, task.PHI0] == 2 * np.pi:
        logger.info("Sampling from 2pi")
        Tx_oob[:, task.PHI0] = False
    idx = np.argmax(np.any(Tx_oob, axis=1))

    ############################################################################
    # First, visualize the trajectory in phase.
    bb_V_nom = bb_V_noms["altpitch"]
    _, bb_Xs, bb_Ys = task.get_contour_x0(0)
    fig, ax = plt.subplots(layout="constrained")
    cs0 = ax.contourf(bb_Xs, bb_Ys, bb_V_nom, norm=CenteredNorm(), levels=11, cmap="RdBu_r", alpha=0.8)
    cs1 = ax.contour(bb_Xs, bb_Ys, bb_V_nom, levels=[0.0], colors=[PlotStyle.ZeroColor], linewidths=[0.8], alpha=0.8)

    ax.plot(T_x[:, task.H0], T_x[:, task.THETA0], color="C2", lw=0.8)
    ax.plot(T_x[0, task.H0], T_x[0, task.THETA0], color="C2", marker="s", ms=1.0)

    ax.plot(T_x_nom[:, task.H0], T_x_nom[:, task.THETA0], color="C0", lw=0.8)
    ax.plot(T_x_nom[0, task.H0], T_x_nom[0, task.THETA0], color="C0", marker="s", ms=1.0)

    # Put a marker at the point where we go out of training bounds.
    ax.plot(T_x[idx, task.H0], T_x[idx, task.THETA0], color="C2", marker="D", ms=1.0, alpha=0.8)
    ax.plot(T_x_nom[idx, task.H0], T_x_nom[idx, task.THETA0], color="C0", marker="D", ms=1.0, alpha=0.8)

    cbar = fig.colorbar(cs0)
    cbar.add_lines(cs1)
    task.plot_altpitch(ax)

    lines = [lline("C2"), lline("C0")]
    labels = ["QP", "Nom"]
    ax.legend(lines, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 1.0), fontsize="x-small")

    fig.savefig(plot_dir / f"altpitch{cid}.pdf", bbox_inches="tight")
    plt.close(fig)

    ############################################################################
    fig, ax = plt.subplots(layout="constrained")
    ax.plot(T_x[:, task.PE0], T_x[:, task.PN0], color="C1", zorder=8)
    ax.plot(T_x[0, task.PE0], T_x[0, task.PN0], marker="s", ms=10, color="C1")

    ax.plot(T_x[:, task.PE1], T_x[:, task.PN1], color="C4", zorder=8)
    ax.plot(T_x[0, task.PE1], T_x[0, task.PN1], marker="s", ms=10, color="C4")

    ax.plot(T_x_nom[:, task.PE0], T_x_nom[:, task.PN0], color="C1", ls="--", zorder=8)
    ax.plot(T_x_nom[:, task.PE1], T_x_nom[:, task.PN1], color="C4", ls="--", zorder=8)

    task.plot_pos2d(ax)
    fig.savefig(plot_dir / f"pos2d{cid}.pdf", bbox_inches="tight")
    plt.close(fig)

    ############################################################################
    fig, ax = plt.subplots(layout="constrained")
    ax.plot(T_x[:, task.PE0], T_x[:, task.H0], color="C1", zorder=8)
    ax.plot(T_x[0, task.PE0], T_x[0, task.H0], marker="s", ms=10, color="C1")

    ax.plot(T_x[:, task.PE1], T_x[:, task.H1], color="C4", zorder=8)
    ax.plot(T_x[0, task.PE1], T_x[0, task.H1], marker="s", ms=10, color="C4")

    ax.plot(T_x_nom[:, task.PE0], T_x_nom[:, task.H0], color="C1", ls="--", zorder=8)
    ax.plot(T_x_nom[:, task.PE1], T_x_nom[:, task.H1], color="C4", ls="--", zorder=8)

    task.plot_eastup(ax)
    fig.savefig(plot_dir / f"eastup{cid}.pdf", bbox_inches="tight")
    plt.close(fig)

    ############################################################################
    # Then, visualize trajectory.
    nrows = task.nx + task.nh + 1
    figsize = np.array([6, 1.0 * nrows])
    x_labels, h_labels = task.x_labels, task.h_labels
    fig, axes = plt.subplots(nrows, figsize=figsize, sharex=True, layout="constrained")
    for ii, ax in enumerate(axes[: task.nx]):
        ax.plot(T_t, T_x[:, ii], color="C1", lw=1.0, label="QP")
        ax.plot(T_t, T_x_nom[:, ii], color="C2", lw=1.0, label="Nom")
        ax.set_ylabel(x_labels[ii], rotation=0, ha="right")

    # Get the label of the argmax of Th_h.
    h_max = Th_h.max(axis=0)
    h_argmax = np.argmax(h_max)
    h_argmax_label = h_labels[h_argmax]

    axes[task.nx].set_title("h. qp argmax: {}".format(h_argmax_label))
    axes[task.nx].plot(T_t, T_Vh_pred, color="C1", lw=1.0)
    axes[task.nx].plot(T_t, Th_h.max(axis=1), color="C1", ls="--", lw=1.0)

    axes[task.nx].plot(T_t, T_Vh_pred_nom, color="C2", lw=1.0)
    axes[task.nx].plot(T_t, Th_h_nom.max(1), color="C2", ls="--", lw=1.0)
    axes[task.nx].set_ylabel("h total", rotation=0, ha="right")

    for ii, ax in enumerate(axes[task.nx + 1 :]):
        ax.plot(T_t, Th_Vh_pred[:, ii], color="C1", alpha=0.8, lw=1.0)
        ax.plot(T_t, Th_h[:, ii], color="C1", ls="--", lw=1.0)

        ax.plot(T_t, Th_Vh_pred_nom[:, ii], color="C2", alpha=0.8, lw=1.0)
        ax.plot(T_t, Th_h_nom[:, ii], color="C2", ls="--", lw=1.0)

        ax.axhline(0.0, color="C3", lw=0.5)
        ax.set_ylabel(h_labels[ii], rotation=0, ha="right")

    if idx > 0:
        [ax.axvline(T_t[idx], color="C6", ls="--", alpha=0.8) for ax in axes]

    # Plot constraint boundaries.
    task.plot_boundaries(axes[: task.nx])
    # Plot training boundaries.
    plot_boundaries(axes[: task.nx], task.train_bounds())
    axes[0].legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, 1.0), fontsize="x-small")
    fig.suptitle("h_max nom={}, qp={}".format(Th_h_nom.max(), Th_h.max()))
    fig.savefig(plot_dir / f"altpitch_traj_{cid}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
