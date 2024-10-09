import ipdb
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import typer
from matplotlib.colors import CenteredNorm, Normalize

from clfrl.dyn.f16_gcas import A_BOUNDS, F16GCAS
from clfrl.dyn.sim_cts_pbar import SimCtsPbar
from clfrl.plotting.plot_utils import plot_boundaries
from clfrl.plotting.plotstyle import PlotStyle
from clfrl.utils.easy_npz import EasyNpz
from clfrl.utils.jax_utils import jax2np, jax_jit, jax_use_double, rep_vmap
from clfrl.utils.paths import get_script_plot_dir

app = typer.Typer()


@app.command()
def gen():
    jax_use_double()

    plot_dir = get_script_plot_dir()
    task = F16GCAS()

    setup_idx = 1
    n_pts = 120

    pol = task.nom_pol_pid

    tf = 5.0
    dt = task.dt
    n_steps = int(round(tf / dt))

    bb_x, bb_Xs, bb_Ys = task.get_contour_x0(setup_idx, n_pts)

    sim = SimCtsPbar(task, pol, n_steps, dt, dt0=dt, use_obs=False, use_pid=True, max_steps=n_steps, solver="bosh3")
    bbT_x, bbT_t = jax2np(jax_jit(rep_vmap(sim.rollout_plot, rep=2))(bb_x))
    bbTh_h = jax2np(jax_jit(rep_vmap(task.h_components, rep=3))(bbT_x))
    bbh_Vh = np.max(bbTh_h, axis=2)
    T_t = bbT_t[0, 0]

    # Save results.
    npz_path = plot_dir / "play_alphabeta.npz"
    np.savez(npz_path, T_t=T_t, bb_Xs=bb_Xs, bb_Ys=bb_Ys, bbT_x=bbT_x, bbTh_h=bbTh_h, bbh_Vh=bbh_Vh)

    plot()


@app.command()
def plot():
    plot_dir = get_script_plot_dir()
    task = F16GCAS()

    npz_path = plot_dir / "play_alphabeta.npz"
    npz = EasyNpz(npz_path)
    T_t, bb_Xs, bb_Ys, bbT_x, bbTh_h, bbh_Vh = npz("T_t", "bb_Xs", "bb_Ys", "bbT_x", "bbTh_h", "bbh_Vh")
    bb_Vh = np.max(bbh_Vh, axis=2)

    T = len(T_t)

    bb2_XY = np.stack([bb_Xs, bb_Ys], axis=2)
    nom = np.array([0.0, 0.2])
    # nom = np.array([500.0, 0.125])
    # nom = task.nominal_val_state()[[task.BETA, task.ALPHA]]

    bb_dists = np.linalg.norm(bb2_XY - nom, axis=-1)
    idx = np.unravel_index(bb_dists.argmin(), bb_Xs.shape)
    T_x = bbT_x[idx[0], idx[1]]
    print("idx: ", idx)

    fig, ax = plt.subplots(layout="constrained")
    cs = ax.contourf(bb_Xs, bb_Ys, bb_Vh, levels=13, norm=CenteredNorm(), cmap="RdBu_r", alpha=0.8)
    ax.contour(bb_Xs, bb_Ys, bb_Vh, levels=[0], colors=[PlotStyle.ZeroColor], alpha=0.95, linewidths=1.0)
    ax.plot(T_x[0, task.BETA], T_x[0, task.ALPHA], marker="o", color="C5", ms=1.0, zorder=10)
    task.plot_alphabeta(ax)
    cbar = fig.colorbar(cs, ax=ax, shrink=0.8)
    fig.savefig(plot_dir / "alphabeta.pdf")
    plt.close(fig)

    figsize = np.array([5, 1.5 * task.nx])
    x_labels = task.x_labels
    fig, axes = plt.subplots(task.nx, figsize=figsize, sharex=True, layout="constrained")
    for ii, ax in enumerate(axes):
        ax.plot(T_t, T_x[:, ii], color="C1")
        ax.set_ylabel(x_labels[ii])
    # Plot alpha and beta bounds.
    axes[task.ALPHA].axhline(A_BOUNDS[0], color="C0", linestyle="--")
    axes[task.ALPHA].axhline(A_BOUNDS[1], color="C0", linestyle="--")
    # Plot training boundaries.
    plot_boundaries(axes, task.train_bounds())
    fig.savefig(plot_dir / "alphabeta_traj.pdf")
    plt.close(fig)

    ###################################################
    # Time to collision.
    bbT_h = np.max(bbTh_h, axis=3)
    bb_has_unsafe = bb_Vh > 0.0
    bb_unsafe_idx = np.argmax(bbT_h > 0.0, axis=2)
    bb_unsafe_t = T_t[bb_unsafe_idx]
    bb_unsafe_t = np.ma.array(bb_unsafe_t, mask=~bb_has_unsafe)

    norm = Normalize(vmin=0.0, vmax=bb_unsafe_t.max())

    fig, ax = plt.subplots(layout="constrained")
    cs = ax.contourf(bb_Xs, bb_Ys, bb_unsafe_t, levels=13, norm=norm, alpha=0.8, cmap="rocket")
    ax.plot(T_x[0, task.H], T_x[0, task.THETA], marker="o", color="C0", ms=1.0)
    task.plot_alphabeta(ax)
    cbar = fig.colorbar(cs, ax=ax, shrink=0.8)
    fig.savefig(plot_dir / "alphabeta_ttc.pdf")
    plt.close(fig)
    ###################################################


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
