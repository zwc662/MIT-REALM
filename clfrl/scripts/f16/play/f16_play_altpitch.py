import ipdb
import matplotlib.pyplot as plt
import numpy as np
import typer
from matplotlib.colors import BoundaryNorm, CenteredNorm, Normalize

from clfrl.dyn.f16_gcas import A_BOUNDS, F16GCAS
from clfrl.dyn.sim_cts_pbar import SimCtsPbar
from clfrl.plotting.plot_utils import plot_boundaries
from clfrl.plotting.plotstyle import PlotStyle
from clfrl.utils.easy_npz import EasyNpz
from clfrl.utils.jax_utils import jax2np, jax_jit, jax_use_double, merge01, rep_vmap, unmerge01
from clfrl.utils.path_utils import mkdir
from clfrl.utils.paths import get_script_plot_dir

app = typer.Typer()


@app.command()
def gen():
    jax_use_double()

    plot_dir = mkdir(get_script_plot_dir() / "altpitch")
    task = F16GCAS()

    setup_idx = 0
    n_pts = 60

    pol = task.nom_pol_pid

    tf = 30.0
    dt = task.dt
    n_steps = int(round(tf / dt))

    bb_x, bb_Xs, bb_Ys = task.get_contour_x0(setup_idx, n_pts)

    sim = SimCtsPbar(task, pol, n_steps, dt, dt0=dt, use_obs=False, use_pid=False, max_steps=n_steps, solver="bosh3")
    bbT_x, bbT_t = jax2np(jax_jit(rep_vmap(sim.rollout_plot, rep=2))(bb_x))
    bbTh_h = jax2np(jax_jit(rep_vmap(task.h_components, rep=3))(bbT_x))
    bbh_Vh = np.max(bbTh_h, axis=2)
    T_t = bbT_t[0, 0]

    # Save results.
    npz_path = plot_dir / "play_altpitch.npz"
    np.savez(npz_path, T_t=T_t, bb_Xs=bb_Xs, bb_Ys=bb_Ys, bbT_x=bbT_x, bbTh_h=bbTh_h, bbh_Vh=bbh_Vh)

    plot()


@app.command()
def plot():
    plot_dir = mkdir(get_script_plot_dir() / "altpitch")
    task = F16GCAS()

    npz_path = plot_dir / "play_altpitch.npz"
    npz = EasyNpz(npz_path)
    T_t, bb_Xs, bb_Ys, bbT_x, bbTh_h, bbh_Vh = npz("T_t", "bb_Xs", "bb_Ys", "bbT_x", "bbTh_h", "bbh_Vh")
    bb_Vh = np.max(bbh_Vh, axis=2)

    bb = np.prod(bb_Xs.shape)

    T = len(T_t)

    bb2_XY = np.stack([bb_Xs, bb_Ys], axis=2)
    # nom = np.array([400.0, 0.0])
    # nom = np.array([300.0, 0.02])
    # nom = np.array([500.0, 0.125])
    nom = np.array([650., -0.9])
    # nom = np.array([600., -1.1])
    # nom = np.array([400.0, -0.8])
    # nom = task.nominal_val_state()[[task.H, task.THETA]]

    bb_dists = np.linalg.norm(bb2_XY - nom, axis=-1)
    idx = np.unravel_index(bb_dists.argmin(), bb_Xs.shape)
    T_x = bbT_x[idx[0], idx[1]]
    print("idx: ", idx)
    print(repr(T_x[0]))
    print("------------------")

    fig, ax = plt.subplots(layout="constrained")
    cs = ax.contourf(bb_Xs, bb_Ys, bb_Vh, levels=13, norm=CenteredNorm(), cmap="RdBu_r", alpha=0.8)
    ax.contour(bb_Xs, bb_Ys, bb_Vh, levels=[0], colors=[PlotStyle.ZeroColor], alpha=0.95, linewidths=1.0)

    ax.plot(T_x[:, task.H], T_x[:, task.THETA], color="C1", lw=0.8)
    ax.plot(T_x[0, task.H], T_x[0, task.THETA], marker="o", color="C2", ms=1.0)
    task.plot_altpitch(ax)
    cbar = fig.colorbar(cs, ax=ax, shrink=0.8)
    fig.savefig(plot_dir / "altpitch.pdf")
    plt.close(fig)

    ################################################################################################
    # Compare two trajectories near the boundary of alpha_lb.
    # nom = np.array([600.0, 0.05])
    nom = np.array([300.0, 0.02])
    bb_dists = np.linalg.norm(bb2_XY - nom, axis=-1)
    idx2 = np.unravel_index(bb_dists.argmin(), bb_Xs.shape)
    T_x2 = bbT_x[idx2[0], idx2[1]]
    T_x3 = bbT_x[idx2[0] + 1, idx2[1]]
    T_x4 = bbT_x[idx2[0] + 2, idx2[1]]
    print("idx2: ", idx2)

    ################################################################################################

    ################################################################################################
    # Plot by each h.
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

        ax.plot(T_x2[0, task.H], T_x2[0, task.THETA], marker="o", mec="none", color="C0", ms=0.7, zorder=10)
        ax.plot(T_x3[0, task.H], T_x3[0, task.THETA], marker="o", mec="none", color="C2", ms=0.7, zorder=10)
        ax.plot(T_x4[0, task.H], T_x4[0, task.THETA], marker="o", mec="none", color="C4", ms=0.7, zorder=10)

        fig.colorbar(cs, ax=ax, shrink=0.8)
        ax.set_title(h_labels[ii])
        task.plot_altpitch(ax)
    fig.savefig(plot_dir / "altpitch_components.pdf")
    plt.close(fig)

    ################################################################################################
    # Compare trajectories near the boundary of alpha_lb.

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
    fig.savefig(plot_dir / "altpitch_traj_αboundary.pdf")
    plt.close(fig)

    ################################################################################################

    figsize = np.array([5, 1.5 * task.nx])
    x_labels = task.x_labels
    fig, axes = plt.subplots(task.nx, figsize=figsize, sharex=True, layout="constrained")
    for ii, ax in enumerate(axes):
        ax.plot(T_t, T_x[:, ii], color="C1")
        ax.set_ylabel(x_labels[ii])
    # Plot alpha and beta bounds.
    task.plot_boundaries(axes)
    # Plot training boundaries.
    plot_boundaries(axes, task.train_bounds())
    fig.savefig(plot_dir / "altpitch_traj.pdf")
    plt.close(fig)

    ###################################################
    # Time to collision.
    bbT_h = np.max(bbTh_h, axis=3)
    bb_has_unsafe = bb_Vh > 0.0
    bb_unsafe_idx = np.argmax(bbT_h > 0.0, axis=2)
    bb_unsafe_t = T_t[bb_unsafe_idx]
    bb_unsafe_t_mask = np.ma.array(bb_unsafe_t, mask=~bb_has_unsafe)

    norm = Normalize(vmin=0.0, vmax=bb_unsafe_t_mask.max())

    fig, ax = plt.subplots(layout="constrained")
    cs = ax.contourf(bb_Xs, bb_Ys, bb_unsafe_t_mask, levels=13, norm=norm, alpha=0.8, cmap="rocket")
    ax.plot(T_x[0, task.H], T_x[0, task.THETA], marker="o", color="C0", ms=1.0)
    task.plot_altpitch(ax)
    cbar = fig.colorbar(cs, ax=ax, shrink=0.8)
    fig.savefig(plot_dir / "altpitch_ttc.pdf")
    plt.close(fig)
    ###################################################
    # Which constraint is active at the collision.
    bbh_h_at_crash = unmerge01(merge01(bbTh_h)[np.arange(bb), merge01(bb_unsafe_idx)], bb_Xs.shape)
    bb_argmax_h = np.argmax(bbh_h_at_crash, axis=2)
    bb_argmax_h_mask = np.ma.array(bb_argmax_h, mask=~bb_has_unsafe)

    # boundaries = np.arange(task.nh + 1)
    # norm = BoundaryNorm(boundaries, ncolors=task.nh)

    ax: plt.Axes
    fig, ax = plt.subplots(layout="constrained")
    cs = ax.contourf(bb_Xs, bb_Ys, bb_argmax_h_mask, levels=np.arange(task.nh), alpha=0.8)
    # cs = ax.pcolormesh(bb_Xs, bb_Ys, bb_argmax_h_mask, alpha=0.8, cmap="rocket", norm=norm, shading="auto")
    ax.plot(T_x[0, task.H], T_x[0, task.THETA], marker="o", color="C0", ms=1.0)
    task.plot_altpitch(ax)
    cbar = fig.colorbar(cs, ax=ax, shrink=0.8)
    fig.savefig(plot_dir / "altpitch_argmax.pdf")
    plt.close(fig)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
