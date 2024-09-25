import pathlib

import ipdb
import matplotlib.pyplot as plt
import numpy as np
import typer
from matplotlib.colors import CenteredNorm

from pncbf.dyn.f16_gcas import F16GCAS
from pncbf.ncbf.compute_disc_avoid import AllDiscAvoidTerms
from pncbf.plotting.legend_helpers import lline
from pncbf.plotting.plotstyle import PlotStyle
from pncbf.utils.ckpt_utils import get_id_from_ckpt, get_run_path_from_ckpt
from pncbf.utils.easy_npz import EasyNpz
from pncbf.utils.jax_utils import jax_default_x32
from pncbf.utils.logging import set_logger_format
from pncbf.utils.path_utils import mkdir
from scripts.f16.dbg_f16gcas_nom import gen as gen_nom

app = typer.Typer()


@app.command()
def gen(ckpt1: pathlib.Path, ckpt2: pathlib.Path):
    jax_default_x32()
    set_logger_format()

    task = F16GCAS()
    x0 = task.nominal_val_state()
    x0[task.H] = 650.0
    x0[task.THETA] = -0.75

    gen_nom(ckpt1, do_plot=False, x0=x0)
    gen_nom(ckpt2, do_plot=False, x0=x0)

    plot(ckpt1, ckpt2)


@app.command()
def plot(ckpt1: pathlib.Path, ckpt2: pathlib.Path):
    jax_default_x32()
    set_logger_format()
    task = F16GCAS()
    cid1 = get_id_from_ckpt(ckpt1)
    cid2 = get_id_from_ckpt(ckpt2)

    run_path = get_run_path_from_ckpt(ckpt1)
    npz_dir = mkdir(run_path / "eval/dbg_f16gcas_nom")
    plot_dir = mkdir(run_path / "eval/compare_nom")

    npz_path1 = npz_dir / f"data{cid1}.pkl"
    npz1 = EasyNpz(npz_path1)

    npz_path2 = npz_dir / f"data{cid2}.pkl"
    npz2 = EasyNpz(npz_path2)
    ##########################################################################
    T_t, T_x, Th_h, Th_Vh_pred1, Teh_Vh_tgt1 = npz1("T_t", "T_x", "Th_h", "Th_Vh_pred", "Teh_Vh_tgt")
    T_t, T_x, Th_h, Th_Vh_pred2, Teh_Vh_tgt2 = npz2("T_t", "T_x", "Th_h", "Th_Vh_pred", "Teh_Vh_tgt")

    vterms1: AllDiscAvoidTerms = npz1("vterms")
    vterms_full1: AllDiscAvoidTerms = npz1("vterms_full")

    bb_V_noms = npz1("bb_V_noms")
    bb_V_nom = bb_V_noms["altpitch"]
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
    fig.savefig(plot_dir / f"altpitch{cid1}{cid2}.pdf", bbox_inches="tight")
    plt.close(fig)
    ############################################################################
    h_labels = task.h_labels
    figsize = np.array([6.0, task.nh * 1.0])
    fig, axes = plt.subplots(task.nh, figsize=figsize, layout="constrained")
    for ii, ax in enumerate(axes):
        ax.axhline(0.0, color="0.4", alpha=0.8, lw=0.8)

        ax.plot(T_t, Th_h[:, ii], color="C3", lw=0.5, ls="--", label=r"$h$")

        ax.plot(T_t, Th_Vh_pred1[:, ii], color="C0", lw=1.0)
        ax.plot(T_t, Th_Vh_pred2[:, ii], color="C1", lw=1.0)

        ax.plot(T_t, Teh_Vh_tgt1[:, 0, ii], color="C0", lw=1.0, alpha=0.6, ls="--")
        ax.plot(T_t, Teh_Vh_tgt1[:, 1, ii], color="C0", lw=1.0, alpha=0.6, ls="--")

        ax.plot(T_t, Teh_Vh_tgt2[:, 0, ii], color="C1", lw=1.0, alpha=0.6, ls="--")
        ax.plot(T_t, Teh_Vh_tgt2[:, 1, ii], color="C1", lw=1.0, alpha=0.6, ls="--")

        # ax.plot(T_t[:T_train], vterms.Th_max_lhs[:, ii], color="C4", label="Th_max_lhs")
        # ax.plot(T_t, vterms_full.Th_max_lhs[:, ii], color="C4", ls="-.", lw=0.5, label="Th_max_lhs full")
        # ax.plot(T_t[:T_train], Th_tgts[:, ii], color="C5", label="tgts")
        ax.set_ylabel(h_labels[ii], rotation=0, ha="right")

    lines = [lline("C0"), lline("C1"), lline("C3"), lline("C3", lw=1.0, alpha=0.6, ls="--")]
    labels = [cid1[1:], cid2[1:], "Vh", "tgt"]
    axes[0].legend(lines, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 1.0), fontsize="x-small")
    fig.savefig(plot_dir / f"h_compare{cid1}{cid2}.pdf", bbox_inches="tight")


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
