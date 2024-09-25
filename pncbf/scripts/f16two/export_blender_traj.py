import functools as ft
import pathlib

import ipdb
import matplotlib.pyplot as plt
import numpy as np
import typer
from loguru import logger

import run_config.int_avoid.f16two_cfg
import run_config.nclf.pend_cfg
import run_config.nclf_pol.pend_cfg
from pncbf.dyn.f16_two import F16Two
from pncbf.dyn.sim_cts_pbar import SimCtsPbar
from pncbf.ncbf.int_avoid import IntAvoid
from pncbf.plotting.plot_utils import plot_boundaries
from pncbf.utils.ckpt_utils import get_id_from_ckpt, get_run_path_from_ckpt, load_ckpt_with_step
from pncbf.utils.easy_npz import EasyNpz, save_data
from pncbf.utils.jax_utils import jax2np, jax_default_x32, jax_jit_np, jax_vmap
from pncbf.utils.logging import set_logger_format
from pncbf.utils.path_utils import mkdir

app = typer.Typer()


@app.command()
def gen(ckpt_path: pathlib.Path):
    set_logger_format()
    jax_default_x32()
    task = F16Two()

    run_path = get_run_path_from_ckpt(ckpt_path)
    plot_dir = mkdir(run_path / "eval" / "blender")

    CFG = run_config.int_avoid.f16two_cfg.get(0)
    train_pol = task.nom_pol_pid
    alg: IntAvoid = IntAvoid.create(0, task, CFG.alg_cfg, train_pol)
    alg, ckpt_path = load_ckpt_with_step(alg, ckpt_path)
    logger.info("Loaded ckpt from {}!".format(ckpt_path))
    cid = get_id_from_ckpt(ckpt_path)

    test_pol = task.nom_pol_N0_pid
    V_shift = -(1 - np.exp(-alg.lam * task.max_ttc)) * task.h_min + 1e-3
    logger.info("lam: {}, V_shift: {}".format(alg.lam, V_shift))

    pols = {
        # "HOCBF": ft.partial(task.nom_pol_handcbf, nom_pol=const_pol),
        "Nom Test": test_pol,
        "Nom Train": train_pol,
        "NCBF Const Shift": ft.partial(alg.get_cbf_control_sloped, 2.0, 100.0, nom_pol=test_pol, V_shift=V_shift),
    }

    x0 = task.nominal_val_state()
    # x0[task.PE0] = -2000

    x0[task.PE0] = -2_500
    x0[task.PN0] = 100
    x0[task.PE1] = 0.0
    x0[task.PN1] = 1_000.0
    x0[task.PSI1] = -0.75 * np.pi
    # x0[task.PSI1] = -0.85 * np.pi

    tf = 12.0
    mult = 2
    dt = task.dt / mult
    n_steps = int(round(tf / dt))
    logger.info("n_steps: {}".format(n_steps))

    def rollout(pol, x0):
        sim = SimCtsPbar(
            task, pol, n_steps, dt, dt0=dt, use_obs=False, use_pid=False, max_steps=n_steps, solver="tsit5"
        )
        T_x, T_t = sim.rollout_plot(x0)
        T_u = jax_vmap(pol)(T_x)
        Th_h = jax_vmap(task.h_components)(T_x)
        return T_t[::mult], T_x[::mult], T_u[::mult], Th_h[::mult]

    outs = {}
    for pol_name, pol in pols.items():
        logger.info("Running {}...".format(pol_name))
        outs[pol_name] = jax2np(ft.partial(rollout, pol)(x0))

    # Save an additional one for "Nom" that has the same states but nominal policy for controls
    T_t, T_x, T_u, Th_h = outs["NCBF Const Shift"]
    T_u_nom = jax_jit_np(jax_vmap(test_pol))(T_x)
    outs["Nom"] = T_t, T_x, T_u_nom, Th_h

    # Save as pkl.
    pkl_path = plot_dir / "data.pkl"
    save_data(pkl_path, **outs)

    plot(ckpt_path)


@app.command()
def plot(ckpt_path: pathlib.Path):
    set_logger_format()
    jax_default_x32()
    task = F16Two()

    run_path = get_run_path_from_ckpt(ckpt_path)
    plot_dir = mkdir(run_path / "eval" / "blender")
    pkl_path = plot_dir / "data.pkl"
    npz = EasyNpz(pkl_path)

    logger.info("Loading pkl from {}".format(pkl_path))

    # keys = ["HOCBF", "Nom", "Nom Train", "NCBF Const Shift"]
    # keys = ["NCBF Const Shift"]
    keys = ["NCBF Const Shift", "Nom Test", "Nom Train"]
    data = {k: npz[k] for k in keys}

    ####################################################################################################
    for name, (T_t, T_x, T_u, Th_h) in data.items():
        print("h  {:12}: {:.2f}".format(name, Th_h.max()))

        if name == "NCBF Const Shift":
            print(T_x.shape)
            print("T_x[-1]:")
            print(repr(T_x[-100]))

    ####################################################################################################
    fig, ax = plt.subplots(layout="constrained")
    for name, (T_t, T_x, T_u, Th_h) in data.items():
        ax.plot(T_x[:, task.H0], T_x[:, task.THETA0], label=name, ls="--", alpha=0.7)
    ax.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, 1.0), fontsize="x-small")
    task.plot_altpitch(ax)
    fig.savefig(plot_dir / "altpitch.pdf")
    plt.close(fig)
    ####################################################################################################
    fig, ax = plt.subplots(layout="constrained")
    for ii, (name, (T_t, T_x, T_u, Th_h)) in enumerate(data.items()):
        ax.plot(T_x[:, task.PE0], T_x[:, task.PN0], label=name, color=f"C{ii + 1}", ls="--", alpha=0.7)
        if ii == 0:
            ax.plot(T_x[:, task.PE1], T_x[:, task.PN1], color="C0", ls="--", alpha=0.7)
    ax.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, 1.0), fontsize="x-small")
    ax.set(xlabel="East", ylabel="North", aspect="equal")
    fig.savefig(plot_dir / "pos2d.pdf")
    plt.close(fig)

    ####################################################################################################
    # Just plot the traj.
    nrows = task.nx + task.nh + 1
    figsize = np.array([6, 1.0 * nrows])
    x_labels, h_labels = task.x_labels, task.h_labels
    fig, axes = plt.subplots(nrows, figsize=figsize, sharex=True, layout="constrained")
    for ii, ax in enumerate(axes[: task.nx]):
        for name, (T_t, T_x, T_u, Th_h) in data.items():
            ax.plot(T_t, T_x[:, ii], label=name)
        ax.set_ylabel(x_labels[ii], rotation=0, ha="right")

    argmax = h_labels[data["NCBF Const Shift"][3].max(0).argmax()]
    axes[task.nx].set_title("h. qp argmax: {}".format(argmax))
    for name, (T_t, T_x, T_u, Th_h) in data.items():
        axes[task.nx].plot(T_t, Th_h.max(1), lw=1.0)
    axes[task.nx].set_ylabel("Vh", rotation=0, ha="right")

    for ii, ax in enumerate(axes[task.nx + 1 :]):
        ax.axhline(0.0, color="C3", lw=0.5)
        ax.set_ylabel(h_labels[ii], rotation=0, ha="right")

        for name, (T_t, T_x, T_u, Th_h) in data.items():
            ax.plot(T_t, Th_h[:, ii])

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

    # keys = ["HOCBF", "Nom", "Nom Train", "NCBF Const Shift"]
    keys = ["NCBF Const Shift"]
    data = {k: npz[k] for k in keys}

    T_t = data["NCBF Const Shift"][0]
    # T_t, T_x, T_u, Th_h
    T_x_dict = {k: v[1] for k, v in data.items()}
    T_u_dict = {k: v[2] for k, v in data.items()}

    # Export as npz.
    npz_path = plot_dir / "export.npz"
    np.savez(npz_path, T_t=T_t, **T_x_dict)
    logger.info("Saved to {}!".format(npz_path.absolute()))


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
