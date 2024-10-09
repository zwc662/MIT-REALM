import functools as ft
import pathlib

import ipdb
import jax
import matplotlib.pyplot as plt
import numpy as np
import typer
from loguru import logger
from matplotlib.colors import CenteredNorm

import run_config.int_avoid.doubleintwall_cfg
import run_config.nclf.pend_cfg
import run_config.nclf_pol.pend_cfg
from clfrl.dyn.doubleint_wall import DoubleIntWall
from clfrl.dyn.sim_cts import SimCtsReal
from clfrl.ncbf.compute_disc_avoid import compute_all_disc_avoid_terms
from clfrl.ncbf.int_avoid import IntAvoid
from clfrl.plotting.plotstyle import PlotStyle
from clfrl.utils.ckpt_utils import get_run_path_from_ckpt, load_ckpt
from clfrl.utils.jax_utils import jax2np, jax_jit, jax_vmap, rep_vmap
from clfrl.utils.logging import set_logger_format


def main(ckpt_path: pathlib.Path):
    set_logger_format()
    seed = 0

    task = DoubleIntWall()
    nom_pol = task.nom_pol_osc

    CFG = run_config.int_avoid.doubleintwall_cfg.get(seed)
    alg: IntAvoid = IntAvoid.create(seed, task, CFG.alg_cfg, nom_pol)
    alg = load_ckpt(alg, ckpt_path)
    logger.info("Loaded ckpt from {}!".format(ckpt_path))

    run_path = get_run_path_from_ckpt(ckpt_path)
    plot_dir = run_path / "plots"

    # Contour to get an idea of how it should look.
    bb_x, bb_Xs, bb_Ys = task.get_contour_x0(n_pts=192)
    bbh_Vh = jax2np(jax_jit(rep_vmap(alg.get_Vh, rep=2))(bb_x))

    # Plot how V varies along a trajectory.
    # x0 = np.array([0.5, -1.0])
    x0 = np.array([0.5, -1.4])

    T = 250
    tf = T * task.dt

    sim = SimCtsReal(task, nom_pol, tf, task.dt / 4.0, use_pid=True)
    T_x_nom, T_t_nom, _ = jax2np(jax_jit(sim.rollout_plot)(x0))
    dt_traj = np.diff(T_t_nom).mean()

    Th_Vh = jax2np(jax_jit(jax_vmap(alg.get_Vh))(T_x_nom))
    Th_h = jax2np(jax_jit(jax_vmap(task.h_components))(T_x_nom))
    Th_disc = jax2np(jax_jit(ft.partial(compute_all_disc_avoid_terms, alg.lam, dt_traj))(Th_h)).Th_max_lhs

    def get_Lg_V(state):
        # (nh, nx)
        hx_Vx = jax.jacobian(alg.get_Vh)(state)
        # (nx, nu)
        G = alg.task.G(state)
        # (nh, nu) -> (nh,)
        h_LG_V = (hx_Vx @ G).flatten()
        return h_LG_V

    Th_LGVh = jax2np(jax_jit(jax_vmap(get_Lg_V))(T_x_nom))

    figsize = np.array([task.nh * 4.5, 3 * 2.0])
    fig, axes = plt.subplots(3, task.nh, figsize=figsize, layout="constrained")
    for ii, ax in enumerate(axes[0, :]):
        cs0 = ax.contourf(bb_Xs, bb_Ys, bbh_Vh[:, :, ii], cmap="RdBu_r", levels=31, norm=CenteredNorm())
        cs1 = ax.contour(
            bb_Xs, bb_Ys, bbh_Vh[:, :, ii], levels=[0.0], colors=[PlotStyle.ZeroColor], alpha=0.98, linewidths=1.0
        )
        ax.plot(T_x_nom[:, 0], T_x_nom[:, 1], color="C1", lw=1.0, zorder=10)
        task.plot_phase(ax)
        cbar = fig.colorbar(cs0, ax=axes[:, ii].ravel().tolist(), shrink=0.9)
        cbar.add_lines(cs1)

    for ii, ax in enumerate(axes[1, :]):
        ax.plot(T_t_nom, Th_Vh[:, ii], zorder=4)
        ax.plot(T_t_nom, Th_h[:, ii], color="C2", ls="--")
        ax.plot(T_t_nom, Th_disc[:, ii], color="C4", lw=2.0, alpha=0.8)

    axes[1, 0].set(ylabel=r"$V^h$")

    for ii, ax in enumerate(axes[2, :]):
        ax.plot(T_t_nom, Th_LGVh[:, ii], zorder=4)

    axes[2, 0].set(ylabel=r"$L_G V^h$")

    [axes[0, ii].set_title(task.h_labels[ii]) for ii in range(task.nh)]
    fig.savefig(plot_dir / "V_along_traj.pdf")
    plt.close(fig)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)
