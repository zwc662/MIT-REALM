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
from pncbf.dyn.f16_gcas import A_BOUNDS, F16GCAS
from pncbf.dyn.sim_cts_pbar import SimCtsPbar
from pncbf.ncbf.int_avoid import IntAvoid
from pncbf.plotting.legend_helpers import lline
from pncbf.plotting.plot_utils import plot_boundaries
from pncbf.plotting.plotstyle import PlotStyle
from pncbf.utils.ckpt_utils import get_id_from_ckpt, get_run_path_from_ckpt, load_ckpt_with_step
from pncbf.utils.easy_npz import EasyNpz
from pncbf.utils.jax_utils import jax2np, jax_default_x32, jax_jit, rep_vmap
from pncbf.utils.logging import set_logger_format
from pncbf.utils.path_utils import mkdir

app = typer.Typer()


@app.command()
def gen(ckpt_path: pathlib.Path):
    jax_default_x32()
    set_logger_format()
    task = F16GCAS()

    run_path = get_run_path_from_ckpt(ckpt_path)
    plot_dir = mkdir(run_path / "eval/dbg_f16gcas")

    CFG = run_config.int_avoid.f16gcas_cfg.get(0)
    nom_pol = task.nom_pol_pid
    alg: IntAvoid = IntAvoid.create(0, task, CFG.alg_cfg, nom_pol)
    alg, ckpt_path = load_ckpt_with_step(alg, ckpt_path)
    logger.info("Loaded ckpt from {}!".format(ckpt_path))
    cid = get_id_from_ckpt(ckpt_path)

    tf = 5.0
    dt = task.dt
    n_steps = int(round(tf / dt))

    n_pts = 128

    setup_idx = 0
    bb_x, bb_Xs, bb_Ys = task.get_contour_x0(setup_idx, n_pts)

    pol = ft.partial(alg.get_cbf_control_sloped, 5.0, 100.0)
    sim = SimCtsPbar(task, pol, n_steps, dt, dt0=dt, use_obs=False, use_pid=False, max_steps=n_steps, solver="bosh3")
    bbT_x, bbT_t = jax2np(jax_jit(rep_vmap(sim.rollout_plot, rep=2))(bb_x))
    bbTh_h = jax2np(jax_jit(rep_vmap(task.h_components, rep=3))(bbT_x))
    bbh_Vh = np.max(bbTh_h, axis=2)
    T_t = bbT_t[0, 0]

    bb_Vh_noms = jax2np(task.get_bb_V_noms())
    bb_Vh_nom = bb_Vh_noms["altpitch"]

    # Save results.
    npz_path = plot_dir / f"altpitch_dbg{cid}.npz"
    np.savez(
        npz_path, T_t=T_t, bb_Xs=bb_Xs, bb_Ys=bb_Ys, bbT_x=bbT_x, bbTh_h=bbTh_h, bbh_Vh=bbh_Vh, bb_Vh_nom=bb_Vh_nom
    )

    plot(ckpt_path)


@app.command()
def plot(ckpt_path: pathlib.Path):
    set_logger_format()
    task = F16GCAS()

    cid = get_id_from_ckpt(ckpt_path)
    run_path = get_run_path_from_ckpt(ckpt_path)
    plot_dir = mkdir(run_path / "eval/dbg_f16gcas")
    npz_path = plot_dir / f"altpitch_dbg{cid}.npz"
    npz = EasyNpz(npz_path)
    T_t, bb_Xs, bb_Ys, bbT_x, bbTh_h, bbh_Vh = npz("T_t", "bb_Xs", "bb_Ys", "bbT_x", "bbTh_h", "bbh_Vh")
    bb_Vh_nom = npz("bb_Vh_nom")
    bb_Vh = np.max(bbh_Vh, axis=2)

    bb2_XY = np.stack([bb_Xs, bb_Ys], axis=2)

    nom = np.array([650.0, 0.08])
    bb_dists = np.linalg.norm(bb2_XY - nom, axis=-1)
    idx2 = np.unravel_index(bb_dists.argmin(), bb_Xs.shape)
    T_x2 = bbT_x[idx2[0], idx2[1]]
    T_x3 = bbT_x[idx2[0] + 2, idx2[1]]
    T_x4 = bbT_x[idx2[0] + 4, idx2[1]]
    print("idx2: ", idx2)

    _, bb_Xs_nom, bb_Ys_nom = task.get_contour_x0(0)

    ########################################################
    # Plot CI, by each h.
    h_labels = task.h_labels
    figsize = np.array([10, 6])
    fig, axes = plt.subplots(2, 3, sharex=True, sharey=True, figsize=figsize, layout="constrained")
    axes = axes.flatten()
    for ii, ax in enumerate(axes):
        cs = ax.contourf(bb_Xs, bb_Ys, bbh_Vh[:, :, ii], levels=13, norm=CenteredNorm(), cmap="RdBu_r", alpha=0.8)
        ax.contour(bb_Xs, bb_Ys, bbh_Vh[:, :, ii], levels=[0], colors=[PlotStyle.ZeroColor], alpha=0.8, linewidths=0.4)

        # Overlay the true CI.
        ax.contour(
            bb_Xs, bb_Ys, bb_Vh, levels=[0], colors=["lime"], alpha=0.6, linestyles=["--"], linewidths=0.2, zorder=8
        )

        # Also compare with nominal.

        ax.contour(
            bb_Xs_nom, bb_Ys_nom, bb_Vh_nom, levels=[0], colors=["cyan"], alpha=0.6, linestyles=["-."], linewidths=0.2, zorder=8
        )

        ax.plot(T_x2[0, task.H], T_x2[0, task.THETA], marker="o", mec="none", color="C0", ms=0.7, zorder=10)
        ax.plot(T_x3[0, task.H], T_x3[0, task.THETA], marker="o", mec="none", color="C2", ms=0.7, zorder=10)
        ax.plot(T_x4[0, task.H], T_x4[0, task.THETA], marker="o", mec="none", color="C4", ms=0.7, zorder=10)

        fig.colorbar(cs, ax=ax, shrink=0.8)
        ax.set_title(h_labels[ii])
        task.plot_altpitch(ax)
    fig.savefig(plot_dir / f"altpitch_components_{cid}.pdf")
    plt.close(fig)

    # Plot trajs.
    figsize = np.array([5, 1.5 * task.nx])
    x_labels = task.x_labels
    fig, axes = plt.subplots(task.nx, figsize=figsize, sharex=True, layout="constrained")
    for ii, ax in enumerate(axes):
        ax.plot(T_t, T_x2[:, ii], color="C0")
        ax.plot(T_t, T_x3[:, ii], color="C2")
        ax.plot(T_t, T_x4[:, ii], color="C4")
        ax.set_ylabel(x_labels[ii])
    # Plot alpha and beta bounds.
    axes[task.ALPHA].axhline(A_BOUNDS[0], color="C0", linestyle="--")
    axes[task.ALPHA].axhline(A_BOUNDS[1], color="C0", linestyle="--")
    # Plot training boundaries.
    plot_boundaries(axes, task.train_bounds())
    fig.savefig(plot_dir / f"altpitch_traj_{cid}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
