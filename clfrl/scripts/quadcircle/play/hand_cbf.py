import pickle
import time

import ipdb
import jax
import matplotlib.pyplot as plt
import numpy as np
import typer
from loguru import logger
from matplotlib.animation import FuncAnimation

from clfrl.dyn.quadcircle import QuadCircle
from clfrl.dyn.sim_cts import SimCtsReal
import clfrl.ncbf.min_norm_cbf as cbf_old
from clfrl.qp.min_norm_cbf import min_norm_cbf
from clfrl.plotting.ani_utils import save_anim
from clfrl.utils.easy_npz import EasyNpz
from clfrl.utils.jax_utils import jax2np, jax_jit, jax_use_cpu, jax_use_double, jax_vmap
from clfrl.utils.logging import set_logger_format
from clfrl.utils.paths import get_script_plot_dir

app = typer.Typer()


@app.command()
def gen():
    jax_use_cpu()
    jax_use_double()

    set_logger_format()
    task = QuadCircle()
    plot_dir = get_script_plot_dir()

    x0 = task.get_state()

    tf = 15.0
    # result_dt = 0.01
    result_dt = 0.05
    pol = task.nom_pol_handcbf
    sim = SimCtsReal(task, pol, tf, result_dt, use_pid=False, max_steps=384)
    logger.info("Rollout...")
    t0 = time.perf_counter()
    T_states, T_t, _ = jax2np(jax_jit(sim.rollout_plot)(x0))
    t1 = time.perf_counter()
    logger.info("Rollout... Done! {} s".format(t1 - t0))
    #
    # def get_control_info(state):
    #     return pol(state)
    #
    # T_control = jax2np(jax_jit(jax_vmap(get_control_info))(T_states))
    # data = dict(T_control=T_control)

    def get_control_info(state):
        u_lb, u_ub = task.u_min, task.u_max
        h_B = task.nom_pol_h_B(state)
        hx_Bx = jax.jacobian(task.nom_pol_h_B)(state)
        f, G = task.f(state), task.G(state)

        u_nom = task.nom_pol_vf(state)
        # u_qp, r, (qp_state, qp_mats) = cbf_old.min_norm_cbf(task.hand_cbf_alpha, u_lb, u_ub, h_B, hx_Bx, f, G, u_nom)
        u_qp, r, sol = min_norm_cbf(task.hand_cbf_alpha, u_lb, u_ub, h_B, hx_Bx, f, G, u_nom)
        u_qp_clipped = u_qp.clip(-1, 1)

        # qp_mats2 = cbf_old.min_norm_cbf_qp_mats2(task.hand_cbf_alpha, u_lb, u_ub, h_B, hx_Bx, f, G, u_nom)

        # Compute the constraint for both u_nom and u_qp.
        xdot_qp = f + G @ u_qp_clipped
        xdot_nom = f + G @ u_nom

        h_Vdot_qp = hx_Bx @ xdot_qp
        h_Vdot_nom = hx_Bx @ xdot_nom

        h_constr_rhs = -task.hand_cbf_alpha * h_B
        h_h = task.nom_pol_h_h(state)

        return (
            u_nom,
            u_qp_clipped,
            h_Vdot_nom,
            h_Vdot_qp,
            h_constr_rhs,
            r,
            h_h,
            h_B,
            (u_qp, sol),
        )

    T_unom, T_uqp, Th_Vdot_nom, Th_Vdot_qp, Th_constr_rhs, T_r, Th_h, Th_B, T_dbg = jax2np(
        jax_jit(jax_vmap(get_control_info))(T_states)
    )

    # # Save debug data.
    # T_uqp, T_qpmats, T_qpmats2 = T_dbg
    # pkl_path = plot_dir / "hand_cbf_dbg.pkl"
    # with open(pkl_path, "wb") as f:
    #     data = [T_uqp, T_unom, T_qpmats, T_qpmats2]
    #     pickle.dump(data, f)
    # logger.info(f"Saved dbg npz to {pkl_path}")

    data = dict(
        T_unom=T_unom,
        T_uqp=T_uqp,
        Th_Vdot_nom=Th_Vdot_nom,
        Th_Vdot_qp=Th_Vdot_qp,
        Th_constr_rhs=Th_constr_rhs,
        T_r=T_r,
        Th_h=Th_h,
        Th_B=Th_B,
    )
    npz_path = plot_dir / f"hand_cbf.npz"
    np.savez(npz_path, T_states=T_states, T_t=T_t, **data)

    plot()
    anim()


@app.command()
def plot():
    set_logger_format()
    task = QuadCircle()

    plot_dir = get_script_plot_dir()
    npz_path = plot_dir / f"hand_cbf.npz"
    npz = EasyNpz(npz_path)
    Th_h, Th_B, T_t = npz("Th_h", "Th_B", "T_t")

    h_labels = task.h_labels

    figsize = np.array([4.0, task.nh * 2.0])
    fig, axes = plt.subplots(task.nh, figsize=figsize, layout="constrained")
    for ii, ax in enumerate(axes):
        ax.plot(T_t, Th_B[:, ii], color="C0", zorder=4, label=r"$B$")
        ax.plot(T_t, Th_h[:, ii], color="C2", ls="--", label=r"$h$")
        ylim = np.array(ax.get_ylim())
        ylim = ylim.clip(-5.0, 1.0)
        ax.set_ylim(*ylim)
        ax.set_title(h_labels[ii])
    axes[-1].legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(plot_dir / f"hand_cbf_Vtraj.pdf")
    plt.close(fig)

    ######################################################################################
    # Visualize the QP solve.
    Th_Vdot_nom, Th_Vdot_qp = npz("Th_Vdot_nom", "Th_Vdot_qp")
    Th_constr_rhs, T_r = npz("Th_constr_rhs", "T_r")

    figsize = np.array([4.0, task.nh * 2.0])
    fig, axes = plt.subplots(task.nh, figsize=figsize, layout="constrained")
    for ii, ax in enumerate(axes):
        ax.plot(T_t, Th_Vdot_qp[:, ii], color="C0", alpha=0.8, label=r"$\dot{V}$ qp")
        ylim = np.array(ax.get_ylim())
        ax.plot(T_t, Th_Vdot_nom[:, ii], color="C2", alpha=0.8, label=r"$\dot{V}$ nom")
        ax.plot(T_t, Th_constr_rhs[:, ii], color="C4", ls="--", label=r"$-\alpha V$")
        # ylim = ylim.clip(min=-5.0, max=5.0)
        ax.set_ylim(*ylim)
        ax.set_title(h_labels[ii])
    axes[-1].legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(plot_dir / f"hand_cbf_qp_traj.pdf")
    plt.close(fig)

    ######################################################################################
    # See what is happening to quad1_obs.
    ii = 2

    t_Bzero = np.argmax(Th_B[:, ii] >= 0.0)
    t_hzero = np.argmax(Th_h[:, ii] >= 0.0)

    fig, axes = plt.subplots(2, layout="constrained")
    axes[0].plot(T_t, Th_B[:, ii], color="C0", label=r"$B$")
    axes[0].plot(T_t, Th_h[:, ii], color="C2", label=r"$h$")
    axes[0].set_ylim(-3.0, 2.0)

    axes[1].plot(T_t, Th_Vdot_qp[:, ii], lw=0.5, marker="o", ms=1.5, color="C0", label=r"$\dot{B}$")
    axes[1].plot(T_t, Th_constr_rhs[:, ii], ls="--", color="C4", label="rhs")
    axes[1].set_ylim(-5.0, 5.0)

    [ax.axvline(T_t[t_Bzero], color="C0", ls="--") for ax in axes]
    [ax.axvline(T_t[t_hzero], color="C2", ls="--") for ax in axes]

    axes[0].legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    axes[1].legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.savefig(plot_dir / f"hand_cbf_quad1obs.pdf")
    plt.close(fig)


@app.command()
def anim():
    set_logger_format()
    task = QuadCircle()

    plot_dir = get_script_plot_dir()
    npz_path = plot_dir / f"hand_cbf.npz"
    npz = EasyNpz(npz_path)
    T_states, T_controls, T_t = npz("T_states", "T_uqp", "T_t")

    anim_T = len(T_states)

    fig, ax = plt.subplots(dpi=200)
    task.setup_ax_pos(ax)
    q1_artist, q2_artist, obs_artist = task.get_artists(show_nom=True, show_control=True)
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
        control = T_controls[kk]

        nom_u = task.nom_pol_vf(T_states[kk])
        q1_artist.update_state(q1)
        q1_artist.update_vecs(nom=nom_u[:2], control=control[:2])

        q2_artist.update_state(q2)
        q2_artist.update_vecs(nom=nom_u[2:], control=control[2:])

        obs_artist.update_state(obs)
        text.set_text(f"t={T_t[kk]:.2f} s")
        return [*artists, text]

    fps = 30.0
    spf = 1 / fps
    mspf = 1_000 * spf
    ani = FuncAnimation(fig, update, frames=anim_T, init_func=init_fn, interval=mspf, blit=True)
    ani_path = plot_dir / "hand_cbf.mp4"
    save_anim(ani, ani_path)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
