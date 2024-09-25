import functools as ft
import pathlib

import ipdb
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import typer
from loguru import logger
from matplotlib.animation import FuncAnimation

import pncbf.ncbf.min_norm_cbf as cbf_old
import run_config.int_avoid.quadcircle_cfg
import run_config.int_avoid.segway_cfg
import run_config.nclf.pend_cfg
import run_config.nclf_pol.pend_cfg
from pncbf.dyn.quadcircle import QuadCircle
from pncbf.dyn.sim_cts import SimCtsReal
from pncbf.ncbf.compute_disc_avoid import compute_all_disc_avoid_terms
from pncbf.ncbf.int_avoid import IntAvoid
from pncbf.plotting.ani_utils import save_anim
from pncbf.plotting.plotter import pretty_str
from pncbf.qp.min_norm_cbf import min_norm_cbf
from pncbf.utils.ckpt_utils import get_id_from_ckpt, get_run_path_from_ckpt, load_ckpt, load_ckpt_with_step
from pncbf.utils.easy_npz import EasyNpz
from pncbf.utils.jax_utils import jax2np, jax_default_x32, jax_jit, jax_use_cpu, jax_use_double, jax_vmap
from pncbf.utils.logging import set_logger_format
from pncbf.utils.path_utils import mkdir

app = typer.Typer()


@app.command()
def gen(ckpt_path: pathlib.Path, name: str = ""):
    # jax_default_x32()
    jax_use_double()
    jax_use_cpu()

    set_logger_format()
    seed = 0
    task = QuadCircle()

    run_path = get_run_path_from_ckpt(ckpt_path)
    plot_dir = mkdir(run_path / "eval")

    CFG = run_config.int_avoid.quadcircle_cfg.get(seed)
    nom_pol = task.nom_pol_vf
    alg: IntAvoid = IntAvoid.create(seed, task, CFG.alg_cfg, nom_pol)
    alg, ckpt_path = load_ckpt_with_step(alg, ckpt_path)
    logger.info("Loaded ckpt from {}!".format(ckpt_path))
    cid = get_id_from_ckpt(ckpt_path)

    x0 = task.get_state()
    # x0[task.POS_QUAD2] = np.array([3.2, 1.0])

    # tf = 15.0
    tf = 30.0
    result_dt = 0.05
    n_steps = int(round(tf / result_dt))
    # alpha_safe, alpha_unsafe = 10.0, 50.0
    # alpha_safe, alpha_unsafe = 0.1, 50.0
    # alpha_safe, alpha_unsafe = 0.5, 50.0
    # alpha_safe, alpha_unsafe = 1.0, 50.0
    alpha_safe, alpha_unsafe = task.get_cbf_alphas()

    pol = ft.partial(alg.get_cbf_control_sloped, alpha_safe, alpha_unsafe)
    sim = SimCtsReal(task, pol, tf, result_dt, use_pid=False, max_steps=n_steps + 5)
    logger.info("Rollout...")
    T_states, T_t, _ = jax2np(jax_jit(sim.rollout_plot)(x0))

    logger.info("Computing controls... Also show nominal policy, and why it is not safe...?")

    def get_control_info(state):
        u_lb, u_ub = task.u_min, task.u_max
        h_Vh = alg.get_Vh(state)
        hx_Vx = jax.jacobian(alg.get_Vh)(state)
        f, G = task.f(state), task.G(state)

        is_safe = jnp.all(h_Vh < 0)
        alpha = jnp.where(is_safe, alpha_safe, alpha_unsafe)

        u_nom = task.nom_pol_vf(state)
        # u_qp, r, (qp_state, qp_mats) = cbf_old.min_norm_cbf(alpha, u_lb, u_ub, h_Vh, hx_Vx, f, G, u_nom)
        u_qp, r, sol = min_norm_cbf(alpha, u_lb, u_ub, h_Vh, hx_Vx, f, G, u_nom)
        u_qp = u_qp.clip(-1, 1)

        # Compute the constraint for both u_nom and u_qp.
        xdot_qp = f + G @ u_qp
        xdot_nom = f + G @ u_nom

        h_Vdot_qp = hx_Vx @ xdot_qp
        h_Vdot_nom = hx_Vx @ xdot_nom

        h_constr_rhs = -alpha * h_Vh

        return u_nom, u_qp, h_Vdot_nom, h_Vdot_qp, h_constr_rhs, r

    # T_controls = jax2np(jax_jit(jax_vmap(pol))(T_states))
    Tu_nom, Tu_qp, Th_Vdot_nom, Th_Vdot_qp, Th_constr_rhs, T_r = jax2np(jax_jit(jax_vmap(get_control_info))(T_states))
    data = dict(
        Tu_nom=Tu_nom, Tu_qp=Tu_qp, Th_Vdot_nom=Th_Vdot_nom, Th_Vdot_qp=Th_Vdot_qp, Th_constr_rhs=Th_constr_rhs, T_r=T_r
    )

    Th_h = jax2np(jax_jit(jax_vmap(task.h_components))(T_states))
    Th_Vh = jax2np(jax_jit(jax_vmap(alg.get_Vh))(T_states))
    Th_Vh_disc = jax_jit(ft.partial(compute_all_disc_avoid_terms, alg.lam, result_dt))(Th_h).Th_max_lhs

    npz_path = plot_dir / f"animdata{name}{cid}.npz"
    np.savez(npz_path, T_states=T_states, Th_h=Th_h, Th_Vh=Th_Vh, Th_Vh_disc=Th_Vh_disc, T_t=T_t, **data)

    plot(ckpt_path, name=name)
    anim(ckpt_path, name=name)


@app.command()
def plot(ckpt_path: pathlib.Path, name: str = ""):
    set_logger_format()
    task = QuadCircle()

    run_path = get_run_path_from_ckpt(ckpt_path)
    plot_dir = run_path / "eval"
    cid = get_id_from_ckpt(ckpt_path)
    npz_path = plot_dir / f"animdata{name}{cid}.npz"
    npz = EasyNpz(npz_path)
    Th_h, Th_Vh, Th_Vh_disc, T_t = npz("Th_h", "Th_Vh", "Th_Vh_disc", "T_t")

    h_labels = task.h_labels

    figsize = np.array([task.nh * 4.5, 3 * 2.0])
    fig, axes = plt.subplots(task.nh, figsize=figsize, layout="constrained")
    for ii, ax in enumerate(axes):
        ax.plot(T_t, Th_Vh[:, ii], color="C0", zorder=4, label=r"$V^h$")
        ax.plot(T_t, Th_h[:, ii], color="C2", ls="--", label=r"$h$")
        ax.plot(T_t, Th_Vh_disc[:, ii], color="C4", lw=2.0, alpha=0.8, label="disc")
        hmin, hmax = Th_h[:, ii].min(), Th_h[:, ii].max()
        ax.set_title("{}   h∈[{}, {}]".format(h_labels[ii], pretty_str(hmin), pretty_str(hmax)))
    axes[-1].legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(plot_dir / f"Vtraj{cid}.pdf")
    plt.close(fig)

    ######################################################################################
    # Visualize the QP solve.
    Tu_nom, Tu_qp, Th_Vdot_nom, Th_Vdot_qp = npz("Tu_nom", "Tu_qp", "Th_Vdot_nom", "Th_Vdot_qp")
    Th_constr_rhs, T_r = npz("Th_constr_rhs", "T_r")

    figsize = np.array([task.nh * 4.5, 3 * 2.0])
    fig, axes = plt.subplots(task.nh, figsize=figsize, layout="constrained")
    for ii, ax in enumerate(axes):
        ax.plot(T_t, Th_Vdot_nom[:, ii], color="C2", alpha=0.8, label=r"$\dot{V}$ nom")
        ax.plot(T_t, Th_Vdot_qp[:, ii], color="C0", alpha=0.8, label=r"$\dot{V}$ qp")
        ylim = np.array(ax.get_ylim())
        ax.plot(T_t, Th_constr_rhs[:, ii], color="C4", ls="--", label=r"$-\alpha V$")
        ylim = ylim.clip(min=-1.0, max=1.0)
        ax.set_ylim(*ylim)
        ax.set_title(h_labels[ii])
    axes[-1].legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(plot_dir / f"qp_traj{cid}.pdf")
    plt.close(fig)

    ######################################################################################
    # Visualize vel for the quads
    T_states = npz("T_states")

    train_bounds = task.train_bounds()

    fig, ax = plt.subplots(layout="constrained")
    ax.plot(T_t, T_states[:, task.VX1], label=r"$v_x$1")
    ax.plot(T_t, T_states[:, task.VY1], label=r"$v_y$1")
    ax.plot(T_t, T_states[:, task.VX2], label=r"$v_x$2")
    ax.plot(T_t, T_states[:, task.VY2], label=r"$v_y$2")
    ax.axhline(train_bounds[0, task.VX1], color="C0", ls="--")
    ax.axhline(train_bounds[1, task.VX1], color="C0", ls="--")
    ax.set(ylabel="Velocity", xlabel="Time (s)")
    fig.savefig(plot_dir / f"v_bounds{cid}.pdf")
    plt.close(fig)


@app.command()
def anim(ckpt_path: pathlib.Path, time: float = None, name: str = ""):
    set_logger_format()
    task = QuadCircle()

    run_path = get_run_path_from_ckpt(ckpt_path)
    plot_dir = run_path / "eval"
    cid = get_id_from_ckpt(ckpt_path)
    npz_path = plot_dir / f"animdata{name}{cid}.npz"
    npz = EasyNpz(npz_path)
    T_states, T_controls, T_t = npz("T_states", "Tu_qp", "T_t")

    if time is None:
        time = T_t[-1]
    T_t_crop = T_t[T_t <= time]
    anim_T = len(T_t_crop)
    T_states, T_controls = T_states[:anim_T], T_controls[:anim_T]

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
    ani_path = plot_dir / f"anim{name}{cid}.mp4"
    save_anim(ani, ani_path)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
