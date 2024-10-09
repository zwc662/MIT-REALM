import functools as ft
import pathlib

import ipdb
import jax
import matplotlib.pyplot as plt
import numpy as np
import typer
from loguru import logger

import run_config.avoid_fixed.doubleintwall_cfg
import run_config.nclf.pend_cfg
import run_config.nclf_pol.pend_cfg
from clfrl.dyn.doubleint_wall import DoubleIntWall
from clfrl.ncbf.avoid_fixed import AvoidFixed
from clfrl.plotting.contour_utils import centered_norm
from clfrl.utils.ckpt_utils import get_run_path_from_ckpt, load_ckpt
from clfrl.utils.jax_utils import jax2np, jax_jit, rep_vmap
from clfrl.utils.logging import set_logger_format


def main(ckpt_path: pathlib.Path):
    set_logger_format()
    seed = 0

    task = DoubleIntWall()
    nom_pol = task.nom_pol_osc

    CFG = run_config.avoid_fixed.doubleintwall_cfg.get(seed)
    alg: AvoidFixed = AvoidFixed.create(seed, task, CFG.alg_cfg, nom_pol)
    alg = load_ckpt(alg, ckpt_path)
    logger.info("Loaded ckpt from {}!".format(ckpt_path))

    run_path = get_run_path_from_ckpt(ckpt_path)
    plot_dir = run_path / "plots"

    # Eval L_G V.
    def get_Lg_V(state):
        # (nh, nx)
        hx_Vx = jax.jacobian(ft.partial(alg.get_Vh, alg.V.params))(state)
        # (nx, nu)
        G = alg.task.G(state)
        # (nh, nu) -> (nh,)
        h_LG_V = (hx_Vx @ G).flatten()
        return h_LG_V

    bb_x, bb_Xs, bb_Ys = task.get_contour_x0(n_pts=192)
    bbh_LGV = jax2np(jax_jit(rep_vmap(get_Lg_V, rep=2))(bb_x))

    # vmin, vmax = bbh_LGV.min(), bbh_LGV.max()
    vmin, vmax = -1.2, 1.2
    norm = centered_norm(vmin, vmax)
    levels = np.linspace(-norm.halfrange, norm.halfrange, 35)

    figsize = np.array([task.nh * 3, 3])
    fig, axes = plt.subplots(1, task.nh, figsize=figsize, layout="constrained")
    for ii, ax in enumerate(axes):
        cs0 = ax.contourf(bb_Xs, bb_Ys, bbh_LGV[:, :, ii], levels=levels, norm=norm, cmap="RdBu_r", alpha=0.9)
        task.plot_phase(ax)
        ax.set_title(task.h_labels[ii])
    fig.colorbar(cs0, ax=axes.ravel().tolist(), shrink=0.9)
    fig.suptitle("LG V")
    fig.savefig(plot_dir / "LG_V.pdf")
    plt.close(fig)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)
