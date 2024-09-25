import functools as ft
import pathlib
import pickle

import ipdb
import jax
import matplotlib.pyplot as plt
import numpy as np
import typer
from loguru import logger
from matplotlib.colors import CenteredNorm

import run_config.int_avoid.f16gcas_cfg
import run_config.nclf.pend_cfg
import run_config.nclf_pol.pend_cfg
from pncbf.dyn.f16_gcas import A_BOUNDS, B_BOUNDS, F16GCAS
from pncbf.dyn.sim_cts_pbar import SimCtsPbar
from pncbf.ncbf.compute_disc_avoid import AllDiscAvoidTerms, compute_all_disc_avoid_terms
from pncbf.ncbf.int_avoid import IntAvoid
from pncbf.plotting.legend_helpers import lline
from pncbf.plotting.plot_utils import plot_boundaries
from pncbf.plotting.plotstyle import PlotStyle
from pncbf.utils.ckpt_utils import get_id_from_ckpt, get_run_path_from_ckpt, load_ckpt_with_step
from pncbf.utils.easy_npz import EasyNpz, save_data
from pncbf.utils.jax_utils import jax2np, jax_default_x32, jax_jit, jax_vmap, rep_vmap
from pncbf.utils.logging import set_logger_format
from pncbf.utils.path_utils import mkdir

app = typer.Typer()


@app.command()
def gen(ckpt_path: pathlib.Path):
    jax_default_x32()
    set_logger_format()
    task = F16GCAS()

    run_path = get_run_path_from_ckpt(ckpt_path)
    plot_dir = mkdir(run_path / "eval/dbg_f16gcas_qp")

    CFG = run_config.int_avoid.f16gcas_cfg.get(0)
    # nom_pol = task.nom_pol_pid
    nom_pol = task.nom_pol_const
    alg: IntAvoid = IntAvoid.create(0, task, CFG.alg_cfg, nom_pol)
    alg, ckpt_path = load_ckpt_with_step(alg, ckpt_path)
    logger.info("Loaded ckpt from {}!".format(ckpt_path))
    cid = get_id_from_ckpt(ckpt_path)

    # tf = 10.0
    # tf = 30.0
    tf = 8.0
    dt = task.dt / 4
    n_steps = int(round(tf / dt))

    x0 = task.nominal_val_state()
    # x0[task.H] = 600
    # x0[task.THETA] = 0.2
    # x0[task.H] = 500
    # x0[task.THETA] = -0.5
    # x0[task.H] = 100
    # x0[task.THETA] = 0.45
    x0[task.H] = 550
    x0[task.THETA] = -0.1

    pol = task.nom_pol_pid
    sim = SimCtsPbar(task, pol, n_steps, dt, dt0=dt, use_obs=False, use_pid=False, max_steps=n_steps, solver="bosh3")
    T_x_nom, T_t_nom = jax2np(jax_jit(sim.rollout_plot)(x0))

    Th_h_nom = jax2np(jax_vmap(task.h_components)(T_x_nom))
    Th_Vh_pred_nom = jax2np(jax_jit(jax_vmap(alg.get_Vh))(T_x_nom))
    Teh_Vh_tgt_nom = jax2np(jax_jit(jax_vmap(alg.get_eVh_tgt))(T_x_nom))

    #####################################################################

    alpha_safe = np.full(task.nh, 5.0)
    # # Pay extra attention to floor.
    # alpha_safe[0] = 4.0
    alpha_unsafe = np.full(task.nh, 20.0)
    pol = ft.partial(alg.get_cbf_control_sloped, alpha_safe, alpha_unsafe)
    sim = SimCtsPbar(task, pol, n_steps, dt, dt0=dt, use_obs=False, use_pid=False, max_steps=n_steps, solver="bosh3")
    T_x, T_t = jax2np(jax_jit(sim.rollout_plot)(x0))

    Th_h = jax2np(jax_vmap(task.h_components)(T_x))
    Th_Vh_pred = jax2np(jax_jit(jax_vmap(alg.get_Vh))(T_x))
    Teh_Vh_tgt = jax2np(jax_jit(jax_vmap(alg.get_eVh_tgt))(T_x))

    # For better visualization purposes, compute the nominal Vh.
    bb_V_noms = jax2np(task.get_bb_V_noms())

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
    task = F16GCAS()

    cid = get_id_from_ckpt(ckpt_path)

    run_path = get_run_path_from_ckpt(ckpt_path)
    plot_dir = mkdir(run_path / "eval/dbg_f16gcas_qp")
    npz_path = plot_dir / f"data{cid}.pkl"
    npz = EasyNpz(npz_path)

    T_t, T_x, T_x_nom = npz("T_t", "T_x", "T_x_nom")
    Th_Vh_pred, Th_Vh_pred_nom = npz("Th_Vh_pred", "Th_Vh_pred_nom")
    bb_V_noms = npz("bb_V_noms")

    T_Vh_pred = np.max(Th_Vh_pred, axis=1)
    T_Vh_pred_nom = np.max(Th_Vh_pred_nom, axis=1)

    # Store the first index where we go out of the training bounds.
    train_bounds = task.train_bounds()
    Tx_oob = (T_x < train_bounds[0]) | (T_x > train_bounds[1])
    # positions and heading angle do not count.
    Tx_oob[:, [task.PN, task.PE, task.PSI]] = False
    if train_bounds[1, task.PHI] - train_bounds[0, task.PHI] == 2 * np.pi:
        logger.info("Sampling from 2pi")
        Tx_oob[:, task.PHI] = False
    idx = np.argmax(np.any(Tx_oob, axis=1))

    ############################################################################
    # First, visualize the trajectory in phase.
    bb_V_nom = bb_V_noms["altpitch"]
    _, bb_Xs, bb_Ys = task.get_contour_x0(0)
    fig, ax = plt.subplots(layout="constrained")
    cs0 = ax.contourf(bb_Xs, bb_Ys, bb_V_nom, norm=CenteredNorm(), levels=11, cmap="RdBu_r", alpha=0.8)
    cs1 = ax.contour(bb_Xs, bb_Ys, bb_V_nom, levels=[0.0], colors=[PlotStyle.ZeroColor], linewidths=[0.8], alpha=0.8)

    ax.plot(T_x[:, task.H], T_x[:, task.THETA], color="C2", lw=0.8)
    ax.plot(T_x[0, task.H], T_x[0, task.THETA], color="C2", marker="s", ms=1.0)

    ax.plot(T_x_nom[:, task.H], T_x_nom[:, task.THETA], color="C0", lw=0.8)
    ax.plot(T_x_nom[0, task.H], T_x_nom[0, task.THETA], color="C0", marker="s", ms=1.0)

    # Put a marker at the point where we go out of training bounds.
    ax.plot(T_x[idx, task.H], T_x[idx, task.THETA], color="C2", marker="D", ms=1.0, alpha=0.8)
    ax.plot(T_x_nom[idx, task.H], T_x_nom[idx, task.THETA], color="C0", marker="D", ms=1.0, alpha=0.8)

    cbar = fig.colorbar(cs0)
    cbar.add_lines(cs1)
    task.plot_altpitch(ax)

    lines = [lline("C2"), lline("C0")]
    labels = ["QP", "Nom"]
    ax.legend(lines, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 1.0), fontsize="x-small")

    fig.savefig(plot_dir / f"altpitch{cid}.pdf", bbox_inches="tight")
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

    axes[task.nx].plot(T_t, T_Vh_pred, color="C1", lw=1.0)
    axes[task.nx].plot(T_t, T_Vh_pred_nom, color="C2", lw=1.0)
    axes[task.nx].set_ylabel("Vh", rotation=0, ha="right")

    for ii, ax in enumerate(axes[task.nx + 1 :]):
        ax.plot(T_t, Th_Vh_pred[:, ii], color="C1", lw=1.0)
        ax.plot(T_t, Th_Vh_pred_nom[:, ii], color="C2", lw=1.0)
        ax.axhline(0.0, color="C3", lw=0.5)
        ax.set_ylabel(h_labels[ii], rotation=0, ha="right")

    if idx > 0:
        [ax.axvline(T_t[idx], color="C6", ls="--", alpha=0.8) for ax in axes]

    # Plot constraint boundaries.
    task.plot_boundaries(axes[: task.nx])
    # Plot training boundaries.
    plot_boundaries(axes[: task.nx], task.train_bounds())

    axes[0].legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, 1.0), fontsize="x-small")
    fig.savefig(plot_dir / f"altpitch_traj_{cid}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
