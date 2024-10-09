import pathlib

import ipdb
import matplotlib.pyplot as plt
import numpy as np
import typer
from loguru import logger
from matplotlib.colors import CenteredNorm

import run_config.int_avoid.f16gcas_cfg
import run_config.nclf.pend_cfg
import run_config.nclf_pol.pend_cfg
from clfrl.dyn.f16_gcas import F16GCAS
from clfrl.dyn.sim_cts_pbar import SimCtsPbar
from clfrl.ncbf.compute_disc_avoid import AllDiscAvoidTerms, compute_all_disc_avoid_terms
from clfrl.ncbf.int_avoid import IntAvoid
from clfrl.plotting.plotstyle import PlotStyle
from clfrl.utils.ckpt_utils import get_id_from_ckpt, get_run_path_from_ckpt, load_ckpt_with_step
from clfrl.utils.easy_npz import EasyNpz, save_data
from clfrl.utils.jax_utils import jax2np, jax_default_x32, jax_jit, jax_vmap
from clfrl.utils.logging import set_logger_format
from clfrl.utils.path_utils import mkdir

app = typer.Typer()


@app.command()
def gen(ckpt_path: pathlib.Path, plot: bool = True):
    jax_default_x32()
    set_logger_format()
    task = F16GCAS()

    run_path = get_run_path_from_ckpt(ckpt_path)
    plot_dir = mkdir(run_path / "eval/dbg_f16gcas_nom")

    CFG = run_config.int_avoid.f16gcas_cfg.get(0)
    nom_pol = task.nom_pol_pid
    alg: IntAvoid = IntAvoid.create(0, task, CFG.alg_cfg, nom_pol)
    alg, ckpt_path = load_ckpt_with_step(alg, ckpt_path)
    logger.info("Loaded ckpt from {}!".format(ckpt_path))
    cid = get_id_from_ckpt(ckpt_path)

    tf = 10.0
    dt = task.dt
    n_steps = int(round(tf / dt))

    x0 = task.nominal_val_state()
    x0[task.H] = 650
    x0[task.THETA] = 0.1

    pol = task.nom_pol_pid
    sim = SimCtsPbar(task, pol, n_steps, dt, dt0=dt, use_obs=False, use_pid=False, max_steps=n_steps, solver="bosh3")
    T_x, T_t = jax2np(jax_jit(sim.rollout_plot)(x0))

    Th_h = jax2np(jax_vmap(task.h_components)(T_x))
    Th_Vh_pred = jax2np(jax_jit(jax_vmap(alg.get_Vh))(T_x))
    Teh_Vh_tgt = jax2np(jax_jit(jax_vmap(alg.get_eVh_tgt))(T_x))

    # For better visualization purposes, compute the nominal Vh.
    bb_V_noms = jax2np(task.get_bb_V_noms())

    # Also, compute value function terms.
    Th_h_rollout = Th_h[: alg.train_cfg.rollout_T + 1, :]
    vterms = jax2np(compute_all_disc_avoid_terms(alg.lam, dt, Th_h_rollout))

    vterms_full = jax2np(compute_all_disc_avoid_terms(alg.lam, dt, Th_h))

    # Save results.
    pkl_path = plot_dir / f"data{cid}.pkl"
    data = dict(
        T_t=T_t,
        T_x=T_x,
        Th_h=Th_h,
        Th_Vh_pred=Th_Vh_pred,
        Teh_Vh_tgt=Teh_Vh_tgt,
        vterms=vterms,
        vterms_full=vterms_full,
        bb_V_noms=bb_V_noms,
    )
    save_data(pkl_path, **data)

    if plot:
        plot(ckpt_path)


@app.command()
def plot(ckpt_path: pathlib.Path):
    jax_default_x32()
    set_logger_format()
    task = F16GCAS()

    cid = get_id_from_ckpt(ckpt_path)

    run_path = get_run_path_from_ckpt(ckpt_path)
    plot_dir = mkdir(run_path / "eval/dbg_f16gcas_nom")
    npz_path = plot_dir / f"data{cid}.pkl"
    npz = EasyNpz(npz_path)

    T_t, T_x, Th_h, Th_Vh_pred, Teh_Vh_tgt = npz("T_t", "T_x", "Th_h", "Th_Vh_pred", "Teh_Vh_tgt")
    vterms: AllDiscAvoidTerms = npz("vterms")
    vterms_full: AllDiscAvoidTerms = npz("vterms_full")
    bb_V_noms = npz("bb_V_noms")
    T_train = len(vterms.Th_max_lhs)

    Th_Vh_tgt = np.mean(Teh_Vh_tgt, axis=1)
    Th_Vh_tgt_ = Th_Vh_tgt[1 : T_train + 1, :]
    Th_tgts = np.maximum(vterms.Th_max_lhs, vterms.Th_disc_int_rhs + vterms.T_discount_rhs[:, None] * Th_Vh_tgt_)

    ############################################################################
    # First, visualize the trajectory in phase.
    bb_V_nom = bb_V_noms["altpitch"]
    _, bb_Xs, bb_Ys = task.get_contour_x0(0)
    fig, ax = plt.subplots(layout="constrained")
    cs0 = ax.contourf(bb_Xs, bb_Ys, bb_V_nom, norm=CenteredNorm(), levels=11, cmap="RdBu_r", alpha=0.8)
    cs1 = ax.contour(bb_Xs, bb_Ys, bb_V_nom, levels=[0.0], colors=[PlotStyle.ZeroColor], linewidths=[0.8], alpha=0.8)
    ax.plot(T_x[:, task.H], T_x[:, task.THETA], color="C2", lw=0.8)
    ax.plot(T_x[0, task.H], T_x[0, task.THETA], color="C2", marker="s", ms=1.0)
    cbar = fig.colorbar(cs0)
    cbar.add_lines(cs1)
    task.plot_altpitch(ax)
    fig.savefig(plot_dir / f"altpitch{cid}.pdf", bbox_inches="tight")
    plt.close(fig)

    ############################################################################
    h_labels = task.h_labels
    figsize = np.array([6.0, task.nh * 1.0])
    fig, axes = plt.subplots(task.nh, figsize=figsize, layout="constrained")
    for ii, ax in enumerate(axes):
        ax.plot(T_t, Th_h[:, ii], color="C3", lw=0.5, ls="--", label=r"$h$")
        ax.plot(T_t, Th_Vh_pred[:, ii], color="C1", label=r"$V^h$")

        ax.plot(T_t, Teh_Vh_tgt[:, 0, ii], color="C2", lw=1.0, alpha=0.9, ls="--", label=r"$V^h$ tgt")
        ax.plot(T_t, Teh_Vh_tgt[:, 1, ii], color="C6", lw=1.0, alpha=0.9, ls="--", label=r"$V^h$ tgt")

        ax.plot(T_t[:T_train], vterms.Th_max_lhs[:, ii], color="C4", label="Th_max_lhs")
        ax.plot(T_t, vterms_full.Th_max_lhs[:, ii], color="C4", ls="-.", lw=0.5, label="Th_max_lhs full")
        ax.plot(T_t[:T_train], Th_tgts[:, ii], color="C5", label="tgts")
        ax.set_ylabel(h_labels[ii], rotation=0, ha="right")

    axes[0].legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, 1.0), fontsize="x-small")
    fig.savefig(plot_dir / f"h_traj{cid}.pdf", bbox_inches="tight")


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
