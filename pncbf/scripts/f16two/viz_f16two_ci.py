import functools as ft
import pathlib
import pickle

import ipdb
import jax
import matplotlib.pyplot as plt
import numpy as np
import typer
from loguru import logger
from matplotlib.ticker import MaxNLocator, AutoLocator

import run_config.int_avoid.f16two_cfg
import run_config.nclf.pend_cfg
import run_config.nclf_pol.pend_cfg
from pncbf.dyn.f16_two import F16Two
from pncbf.dyn.sim_cts_pbar import SimCtsPbar
from pncbf.ncbf.int_avoid import IntAvoid
from pncbf.plotting.legend_helpers import lline
from pncbf.utils.ckpt_utils import get_id_from_ckpt, get_run_path_from_ckpt, load_ckpt_with_step
from pncbf.utils.jax_utils import jax2np, jax_default_x32, jax_jit, rep_vmap
from pncbf.utils.logging import set_logger_format
from pncbf.utils.path_utils import mkdir

app = typer.Typer()


@app.command()
def gen(ckpt_path: pathlib.Path):
    set_logger_format()
    jax_default_x32()
    task = F16Two()

    run_path = get_run_path_from_ckpt(ckpt_path)
    plot_dir = mkdir(run_path / "eval")

    CFG = run_config.int_avoid.f16two_cfg.get(0)
    train_pol = task.nom_pol_pid
    alg: IntAvoid = IntAvoid.create(0, task, CFG.alg_cfg, train_pol)
    alg, ckpt_path = load_ckpt_with_step(alg, ckpt_path)
    logger.info("Loaded ckpt from {}!".format(ckpt_path))
    cid = get_id_from_ckpt(ckpt_path)

    # def const_pol(x):
    #     # return 0.5 * (task.u_min + task.u_max)
    #     # alpha = 0.3
    #     alpha = 0.45
    #     return (1 - alpha) * task.u_min + alpha * task.u_max

    test_pol = task.nom_pol_N0_pid

    # lam_new = 0.11687734723091125
    # logger.info("Replacing lam {} -> {}".format(alg.lam, lam_new))
    # alg = alg.replace(_lam=lam_new)

    V_shift = -(1 - np.exp(-alg.lam * task.max_ttc)) * task.h_min
    logger.info("lam: {}, V_shift: {}".format(alg.lam, V_shift))

    pols = {
        "NCBF": ft.partial(alg.get_cbf_control_sloped, 2.0, 100.0, nom_pol=test_pol, V_shift=V_shift),
        "Nom (Train)": train_pol,
        "Nom (Test)": test_pol,
        # "HOCBF": ft.partial(task.nom_pol_handcbf, nom_pol=const_pol),
        # "NCBF": ft.partial(alg.get_cbf_control_sloped, 5.0, 100.0),
        # "NCBF Const": ft.partial(alg.get_cbf_control_sloped, 5.0, 100.0, nom_pol=const_pol),
    }

    # tf = 10.0
    tf = 8.0
    dt = task.dt
    # n_pts = 128
    n_pts = 80
    n_steps = int(round(tf / dt))
    tf = n_steps * dt
    logger.info("n_steps: {}".format(n_steps))

    def get_bb_Vh(pol, bb_x_):
        sim = SimCtsPbar(
            task, pol, n_steps, dt, dt0=dt, use_obs=False, use_pid=False, max_steps=n_steps, solver="bosh3"
        )
        bbT_x, _ = rep_vmap(sim.rollout_plot, rep=2)(bb_x_)
        bbT_h = rep_vmap(task.h, rep=3)(bbT_x)
        bb_h = bbT_h.max(axis=-1)
        return bb_h

    get_bb_Vh_pol = {k: jax_jit(ft.partial(get_bb_Vh, pol)) for k, pol in pols.items()}

    vmap_jit_get_V = jax.jit(rep_vmap(alg.get_V, rep=2))

    ###########################################################
    bb_Vh_dicts = []
    for setup in task.phase2d_setups():
        logger.info(f"Generating {setup.plot_name}...")

        # bb_x, bb_Xs, bb_Ys = task.get_ci_x0(setup=setup.setup_idx, n_pts=n_pts)
        bb_x, bb_Xs, bb_Ys = task.get_contour_x0(setup=setup.setup_idx, n_pts=n_pts)

        bb_Vh_dict = {}
        for pol_name, pol in pols.items():
            logger.info(f"    {pol_name}...")
            bb_h = jax2np(get_bb_Vh_pol[pol_name](bb_x))
            bb_Vh_dict[pol_name] = bb_h

        bb_Vh_dicts.append(bb_Vh_dict)

        # Also add in the predicted CI
        bb_Vh_dict["Pred"] = jax2np(vmap_jit_get_V(bb_x))
        bb_Vh_dict["Pred Shift"] = bb_Vh_dict["Pred"] + V_shift

    pkl_path = plot_dir / f"cidata{cid}.pkl"
    with open(pkl_path, "wb") as f:
        data = [bb_Vh_dicts, n_pts]
        pickle.dump(data, f)

    plot(ckpt_path)


@app.command()
def plot(ckpt_path: pathlib.Path):
    set_logger_format()
    task = F16Two()

    run_path = get_run_path_from_ckpt(ckpt_path)
    plot_dir = mkdir(run_path / "eval")

    if ckpt_path.exists():
        cid = get_id_from_ckpt(ckpt_path)
    else:
        num = int(ckpt_path.name)
        cid = "_{:07d}".format(num)

    ###########################################################
    pkl_path = plot_dir / f"cidata{cid}.pkl"
    logger.info("Loading from {}...".format(pkl_path))
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    logger.info("Loaded!")
    bb_Vh_dicts, n_pts = data
    ###########################################################

    # colors = {"Nom": "C0", "NCBF Const": "C1", "NCBF Const Shift": "C2", "Pred": "C5", "Pred Shift": "C4"}
    colors = {
        "Nom (Train)": "C0",
        "Nom (Test)": "C5",
        # "HOCBF": "C2",
        "NCBF": "C1",
        "Pred Shift": "C4",
    }
    replace = {"Pred Shift": "NCBF Boundary"}

    # colors = {
    #     "HOCBF": "C0",
    #     "NCBF": "C1",
    #     "NCBF Const": "C2",
    #     "NCBF Const Shift": "C6",
    #     "Nom": "C5",
    #     "Pred": "C4",
    #     "Pred Shift": "C2",
    # }
    levels = [0]
    linestyles = ["--"]

    ###########################################################
    figsize = 3.0 * np.array([6.4, 4.8])
    for setup in task.phase2d_setups():
        logger.info(f"Plotting {setup.plot_name}...")

        bb_Vh_dict = bb_Vh_dicts[setup.setup_idx]

        bb_x, bb_Xs, bb_Ys = task.get_contour_x0(setup.setup_idx, n_pts=n_pts)
        # bb_x, bb_Xs, bb_Ys = task.get_ci_x0(setup.setup_idx, n_pts=n_pts)
        ax: plt.Axes
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
        ax.xaxis.major.locator.set_params(nbins=11)
        ax.yaxis.major.locator.set_params(nbins=11)

        for pol_name, bb_Vh in bb_Vh_dict.items():
            if pol_name not in colors:
                continue

            opts = {}
            if pol_name == "NCBF":
                opts = {"hatches": ["//"]}

            color = colors[pol_name]
            ax.contourf(
                bb_Xs, bb_Ys, np.ma.array(bb_Vh <= 0, mask=bb_Vh > 0), colors=[color], **opts, alpha=0.2, zorder=3.5
            )
            ax.contour(bb_Xs, bb_Ys, bb_Vh, levels=levels, linestyles=linestyles, colors=[color], zorder=3.5)
        setup.plot(ax)

        lines = [lline(color) for name, color in colors.items() if name in bb_Vh_dict]
        labels = [name for name in colors.keys() if name in bb_Vh_dict]
        labels = [replace.get(label, label) for label in labels]
        ax.legend(lines, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 1.0), fontsize="x-small")
        fig.savefig(plot_dir / f"ci_{setup.plot_name}{cid}.pdf")
        plt.close(fig)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
