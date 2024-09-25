import functools as ft
import pathlib
import pickle

import ipdb
import jax
import matplotlib.pyplot as plt
import numpy as np
import typer
from loguru import logger

import run_config.int_avoid.f16gcas_cfg
import run_config.nclf.pend_cfg
import run_config.nclf_pol.pend_cfg
from pncbf.dyn.f16_gcas import F16GCAS
from pncbf.dyn.sim_cts_pbar import SimCtsPbar
from pncbf.ncbf.int_avoid import IntAvoid
from pncbf.plotting.legend_helpers import lline
from pncbf.plotting.plot_utils import plot_boundaries
from pncbf.utils.ckpt_utils import get_id_from_ckpt, get_run_path_from_ckpt, load_ckpt_with_step
from pncbf.utils.easy_npz import EasyNpz, save_data
from pncbf.utils.jax_utils import jax2np, jax_default_x32, jax_jit, jax_vmap, rep_vmap
from pncbf.utils.logging import set_logger_format
from pncbf.utils.path_utils import mkdir

app = typer.Typer()


@app.command()
def gen(ckpt_path: pathlib.Path):
    set_logger_format()
    jax_default_x32()
    task = F16GCAS()

    run_path = get_run_path_from_ckpt(ckpt_path)
    plot_dir = mkdir(run_path / "eval" / "blender")

    CFG = run_config.int_avoid.f16gcas_cfg.get(0)
    nom_pol = task.nom_pol_pid
    alg: IntAvoid = IntAvoid.create(0, task, CFG.alg_cfg, nom_pol)
    alg, ckpt_path = load_ckpt_with_step(alg, ckpt_path)
    logger.info("Loaded ckpt from {}!".format(ckpt_path))
    cid = get_id_from_ckpt(ckpt_path)

    def const_pol(x):
        alpha = 0.3
        return (1 - alpha) * task.u_min + alpha * task.u_max

    V_shift = -(1 - np.exp(-alg.lam * task.max_ttc)) * task.h_min

    pols = {
        "HOCBF": ft.partial(task.nom_pol_handcbf, nom_pol=const_pol),
        "Nom": const_pol,
        "Nom Train": nom_pol,
        "NCBF Const Shift": ft.partial(alg.get_cbf_control_sloped, 5.0, 100.0, nom_pol=const_pol, V_shift=V_shift),
    }

    x0 = task.nominal_val_state()

    tf = 12.0
    dt = task.dt
    n_steps = int(round(tf / dt))
    logger.info("n_steps: {}".format(n_steps))

    def rollout(pol, x0):
        sim = SimCtsPbar(
            task, pol, n_steps, dt, dt0=dt, use_obs=False, use_pid=False, max_steps=n_steps, solver="tsit5"
        )
        T_x, T_t = sim.rollout_plot(x0)
        T_u = jax_vmap(pol)(T_x)
        Th_h = jax_vmap(task.h_components)(T_x)
        return T_t, T_x, T_u, Th_h

    outs = {}
    for pol_name, pol in pols.items():
        logger.info("Running {}...".format(pol_name))
        outs[pol_name] = jax2np(ft.partial(rollout, pol)(x0))

    # Save as pkl.
    pkl_path = plot_dir / "data.pkl"
    save_data(pkl_path, **outs)

    plot(ckpt_path)


@app.command()
def plot(ckpt_path: pathlib.Path):
    set_logger_format()
    jax_default_x32()
    task = F16GCAS()

    run_path = get_run_path_from_ckpt(ckpt_path)
    plot_dir = mkdir(run_path / "eval" / "blender")
    pkl_path = plot_dir / "data.pkl"
    npz = EasyNpz(pkl_path)

    keys = ["HOCBF", "Nom", "Nom Train", "NCBF Const Shift"]
    data = {k: npz[k] for k in keys}

    ####################################################################################################
    fig, ax = plt.subplots(layout="constrained")
    for name, (T_t, T_x, T_u, Th_h) in data.items():
        ax.plot(T_x[:, task.H], T_x[:, task.THETA], label=name, ls="--", alpha=0.7)
    ax.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, 1.0), fontsize="x-small")
    task.plot_altpitch(ax)
    fig.savefig(plot_dir / "altpitch.pdf")
    plt.close(fig)

    ####################################################################################################
    # Just plot the traj.
    nrows = task.nx
    figsize = np.array([6, 1.2 * nrows])
    x_labels = task.x_labels
    fig, axes = plt.subplots(nrows, figsize=figsize, sharex=True, layout="constrained")
    for ii, ax in enumerate(axes[: task.nx]):
        for name, (T_t, T_x, T_u, Th_h) in data.items():
            ax.plot(T_t, T_x[:, ii], label=name)
        ax.set_ylabel(x_labels[ii], rotation=0, ha="right")
    axes[0].legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, 1.0), fontsize="x-small")

    # Plot constr bounds.
    task.plot_boundaries(axes[: task.nx])
    # Plot training boundaries.
    plot_boundaries(axes[: task.nx], task.train_bounds())
    fig.savefig(plot_dir / "traj.pdf")
    plt.close(fig)
    ####################################################################################################


@app.command()
def export(ckpt_path: pathlib.Path):
    set_logger_format()

    run_path = get_run_path_from_ckpt(ckpt_path)
    plot_dir = mkdir(run_path / "eval" / "blender")
    pkl_path = plot_dir / "data.pkl"
    npz = EasyNpz(pkl_path)

    keys = ["HOCBF", "Nom", "Nom Train", "NCBF Const Shift"]
    data = {k: npz[k] for k in keys}

    T_t = data["HOCBF"][0]
    # T_t, T_x, T_u, Th_h
    T_x_dict = {k: v[1] for k, v in data.items()}
    T_u_dict = {k: v[2] for k, v in data.items()}

    # Export as npz.
    npz_path = plot_dir / "export.npz"
    np.savez(npz_path, T_t=T_t, **T_x_dict)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
