import pathlib

import ipdb
import matplotlib.pyplot as plt
import numpy as np
import typer
from matplotlib.animation import FuncAnimation

from clfrl.dyn.quadcircle import QuadCircle
from clfrl.dyn.sim_cts import SimCtsReal
from clfrl.plotting.ani_utils import save_anim
from clfrl.plotting.circ_artist import CircArtist
from clfrl.utils.easy_npz import EasyNpz
from clfrl.utils.paths import get_script_plot_dir

app = typer.Typer()


@app.command()
def gen():
    plot_dir = get_script_plot_dir()
    task = QuadCircle()

    tf = 3.0

    x0 = task.nominal_val_state()

    result_dt = 0.05
    sim = SimCtsReal(task, task.nom_pol_vf, tf, result_dt)
    T_states, T_t, _ = sim.rollout_plot(x0)

    npz_path = pathlib.Path(plot_dir / "anim_quadcircle.npz")
    np.savez(npz_path, T_states=T_states, T_t=T_t)

    anim()


@app.command()
def anim():
    plot_dir = get_script_plot_dir()
    task = QuadCircle()

    npz_path = pathlib.Path(plot_dir / "anim_quadcircle.npz")
    npz = EasyNpz(npz_path)
    T_states, T_t = npz("T_states", "T_t")
    anim_T = len(T_states)

    fig, ax = plt.subplots(dpi=200)
    task.setup_ax_pos(ax)

    q1_artist, q2_artist, obs_artist = task.get_artists()
    artists = [q1_artist, q2_artist, obs_artist]
    [ax.add_artist(artist) for artist in artists]
    # Visualize the path that the obstacle will take.
    task.viz_obs_path(ax, T_states[0])
    # Show time and timestep.
    text: plt.Text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    def init_fn() -> list[plt.Artist]:
        return [*artists, text]

    def update(kk: int) -> list[plt.Artist]:
        q1, q2, obs = task.split_state(T_states[kk])
        nom_u = task.nom_pol_vf(T_states[kk])
        q1_artist.update_state(q1)
        q1_artist.update_vecs(nom=nom_u[:2])
        q2_artist.update_state(q2)
        q2_artist.update_vecs(nom=nom_u[2:])
        obs_artist.update_state(obs)
        text.set_text(f"t={T_t[kk]:.2f} s")
        return [*artists, text]

    fps = 30.0
    spf = 1 / fps
    mspf = 1_000 * spf
    ani = FuncAnimation(fig, update, frames=anim_T, init_func=init_fn, interval=mspf, blit=True)
    ani_path = plot_dir / "anim_quadcircle.mp4"
    save_anim(ani, ani_path)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
