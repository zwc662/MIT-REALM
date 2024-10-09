import functools as ft

import ipdb
import matplotlib.pyplot as plt
import numpy as np
import typer
from loguru import logger
from matplotlib.animation import FuncAnimation
from rich.progress import Progress

from clfrl.dyn.segway import Segway
from clfrl.dyn.sim_cts import SimCts, integrate
from clfrl.plotting.segway_artist import SegwayArtist
from clfrl.utils.easy_npz import EasyNpz
from clfrl.utils.jax_utils import jax2np, jax_jit, jax_use_cpu, jax_vmap
from clfrl.utils.path_utils import mkdir
from clfrl.utils.paths import get_script_plot_dir

app = typer.Typer()


@app.command()
def gen():
    plot_dir = mkdir(get_script_plot_dir() / "segway")
    jax_use_cpu()

    task = Segway()

    T = 800
    dt = task.dt

    b_x0 = np.array(
        [
            [0.0, -0.07, 0.0, -3.5],
            [0.0, -0.05, 0.0, -3.5],
            [0.0, 0.42, 0.0, -3.5],
            [0.0, 0.48, 0.0, -3.5],
            [0.0, 0.49, 0.0, -3.5],
        ]
    )

    interp_pts = 4
    sim = SimCts(task, task.nom_pol_lqr, T, interp_pts, pol_dt=dt, use_obs=False)
    bT_x, bT_u, bT_t = jax2np(jax_jit(jax_vmap(sim.rollout_plot))(b_x0))
    np.savez(plot_dir / "bifurcate.npz", bT_x=bT_x, bT_t=bT_t)

    anim()


@app.command()
def anim():
    plot_dir = mkdir(get_script_plot_dir() / "segway")
    segway_params = Segway.Params()

    logger.info("Loading...")
    npz = EasyNpz(plot_dir / "bifurcate.npz")
    bT_t, bT_x = npz("bT_t", "bT_x")
    print(bT_t.shape, bT_x.shape)
    b, T = bT_t.shape
    T_kk = np.arange(T)

    ii = np.array([3, 4, 5])
    bT_t, bT_x = bT_t[ii], bT_x[ii]

    bT_x[0, 590:, :] = bT_x[0, 590, :]
    bT_x[1, 920:, :] = bT_x[1, 920, :]
    bT_x = bT_x[:, :2400]

    anim_every = 8
    bT_t, bT_x, T_kk = bT_t[:, ::anim_every], bT_x[:, ::anim_every], T_kk[::anim_every]
    anim_T = bT_x.shape[1]
    logger.info("Animating every {} frames, {} -> {}".format(anim_every, T, anim_T))

    xmin, xmax = np.min(bT_x[:, :, 0]), np.max(bT_x[:, :, 0])
    xrange = xmax - xmin
    xpad = 0.1 * xrange
    xmin, xmax = xmin - 0.5 * xpad, xmax + 0.5 * xpad
    ypad = 0.1
    yrange = ypad + segway_params.l + ypad

    # Animate.
    fig, ax = plt.subplots(figsize=(xrange + xpad, yrange), dpi=250)
    ax.set_aspect("equal")
    ax.set(xlim=(xmin, xmax), ylim=(-ypad, segway_params.l + ypad))
    # Remoev the y axis.
    ax.get_yaxis().set_visible(False)

    # Draw ground.
    ax.axhline(0, color="k", linewidth=2)

    # Draw segway.
    params = SegwayArtist.Params(pole_length=segway_params.l, wheel_radius=0.2)
    segways = [SegwayArtist(params, pole_color=f"C{ii}") for ii in range(3)]
    [ax.add_artist(a) for a in segways]
    # Show time and timestep.
    text: plt.Text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    def init_fn() -> list[plt.Artist]:
        return [*segways, text]

    def update(kk: int) -> list[plt.Artist]:
        for ii, state in enumerate(bT_x[:, kk]):
            segways[ii].update_state(state)
        text.set_text(f"t={bT_t[0, kk]:.2f} s, kk={T_kk[kk]}")
        return [*segways, text]

    fps = 30.0
    spf = 1 / fps
    mspf = 1_000 * spf
    ani = FuncAnimation(fig, update, frames=anim_T, init_func=init_fn, interval=mspf, blit=True)

    pbar = Progress()
    pbar.start()
    task = pbar.add_task("Animating", total=anim_T)

    def progress_callback(curr_frame: int, total_frames: int):
        pbar.update(task, advance=1)

    ani_path = plot_dir / "bifurcate_345.mp4"
    pbar.log("Saving anim to {}...".format(ani_path))
    ani.save(ani_path, progress_callback=progress_callback)
    pbar.stop()


@app.command()
def anim2():
    plot_dir = mkdir(get_script_plot_dir() / "segway")
    segway_params = Segway.Params()

    logger.info("Loading...")
    npz = EasyNpz(plot_dir / "bifurcate.npz")
    bT_t, bT_x = npz("bT_t", "bT_x")
    print(bT_t.shape, bT_x.shape)

    b, T = bT_t.shape
    for ii in range(b):
        T_t, T_x = bT_t[ii], bT_x[ii]

        # Get first index where theta drops out of [-pi/2, pi/2].
        theta = T_x[:, 1]
        theta_idx = np.argmax(np.abs(theta) > 0.55 * np.pi)
        if not np.any(np.abs(theta) > 0.55 * np.pi):
            theta_idx = len(T_x)

        # Limit to first theta_idx.
        T_t, T_x = T_t[:theta_idx], T_x[:theta_idx]

        T = len(T_x)
        T_kk = np.arange(T)

        anim_every = 10
        T_t, T_x, T_kk = T_t[::anim_every], T_x[::anim_every], T_kk[::anim_every]
        anim_T = len(T_x)
        logger.info("Animating every {} frames, {} -> {}".format(anim_every, T, anim_T))

        # Get bounds.
        xmin, xmax = np.min(T_x[:, 0]), np.max(T_x[:, 0])
        xrange = xmax - xmin
        xpad = 0.1 * xrange
        xmin, xmax = xmin - 0.5 * xpad, xmax + 0.5 * xpad
        ypad = 0.1
        yrange = ypad + segway_params.l + ypad

        # Animate.
        fig, ax = plt.subplots(figsize=(xrange + xpad, yrange), dpi=250)
        ax.set_aspect("equal")
        ax.set(xlim=(xmin, xmax), ylim=(-ypad, segway_params.l + ypad))
        # Remoev the y axis.
        ax.get_yaxis().set_visible(False)

        # Draw ground.
        ax.axhline(0, color="k", linewidth=2)
        # Draw segway.
        params = SegwayArtist.Params(pole_length=segway_params.l, wheel_radius=0.2)
        segway_artist = SegwayArtist(params)
        ax.add_artist(segway_artist)
        # Show time and timestep.
        text: plt.Text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

        def init_fn() -> list[plt.Artist]:
            return [segway_artist, text]

        def update(kk: int) -> list[plt.Artist]:
            state = T_x[kk]
            segway_artist.update_state(state)
            text.set_text(f"t={T_t[kk]:.2f} s, kk={T_kk[kk]}")
            return [segway_artist, text]

        fps = 30.0
        spf = 1 / fps
        mspf = 1_000 * spf
        ani = FuncAnimation(fig, update, frames=anim_T, init_func=init_fn, interval=mspf, blit=True)

        pbar = Progress()
        pbar.start()
        task = pbar.add_task("Animating", total=anim_T)

        def progress_callback(curr_frame: int, total_frames: int):
            pbar.update(task, advance=1)

        ani_path = plot_dir / "bifurcate_{}.mp4".format(ii)
        pbar.log("Saving anim to {}...".format(ani_path))
        ani.save(ani_path, progress_callback=progress_callback)
        pbar.stop()


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
