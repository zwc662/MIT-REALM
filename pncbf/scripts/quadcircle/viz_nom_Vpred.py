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

import run_config.int_avoid.quadcircle_cfg
import run_config.int_avoid.segway_cfg
import run_config.nclf.pend_cfg
import run_config.nclf_pol.pend_cfg
from pncbf.dyn.quadcircle import QuadCircle
from pncbf.dyn.sim_cts import SimCtsReal
from pncbf.ncbf.compute_disc_avoid import compute_all_disc_avoid_terms
from pncbf.ncbf.int_avoid import IntAvoid
from pncbf.plotting.ani_utils import save_anim
from pncbf.plotting.legend_helpers import lline
from pncbf.utils.ckpt_utils import get_id_from_ckpt, get_run_path_from_ckpt, load_ckpt
from pncbf.utils.easy_npz import EasyNpz
from pncbf.utils.jax_utils import jax2np, jax_jit, jax_vmap
from pncbf.utils.logging import set_logger_format
from pncbf.utils.path_utils import mkdir


def main(ckpt_path: pathlib.Path):
    set_logger_format()
    seed = 0
    task = QuadCircle()

    run_path = get_run_path_from_ckpt(ckpt_path)
    plot_dir = mkdir(run_path / "eval")

    CFG = run_config.int_avoid.quadcircle_cfg.get(seed)
    nom_pol = task.nom_pol_vf
    alg: IntAvoid = IntAvoid.create(seed, task, CFG.alg_cfg, nom_pol)
    alg = load_ckpt(alg, ckpt_path)
    logger.info("Loaded ckpt from {}!".format(ckpt_path))
    cid = get_id_from_ckpt(ckpt_path)

    x0 = task.get_state()

    # Rollout using nominal, eval V along traj.
    tf = 10.0
    result_dt = 0.05

    nom_pol = task.nom_pol_vf
    sim = SimCtsReal(task, nom_pol, tf, result_dt, use_pid=True)
    logger.info("Rollout...")
    T_states, T_t, _ = jax2np(jax_jit(sim.rollout_plot)(x0))

    logger.info("Computing V...")

    def get_V(state):
        h_V = alg.get_Vh(state)
        hx_Vx = jax.jacobian(alg.get_Vh)(state)
        f, G = task.f(state), task.G(state)
        u_nom = nom_pol(state)
        h_h = task.h_components(state)

        xdot = f + G @ u_nom
        assert xdot.shape == f.shape == state.shape
        h_Vdot = hx_Vx @ xdot
        h_Vdot_disc = h_Vdot - alg.lam * (h_V - h_h)
        return h_V, h_Vdot, h_Vdot_disc, h_h

    Th_Vh, Th_Vdot, Th_Vdot_disc, Th_h = jax2np(jax_jit(jax_vmap(get_V))(T_states))
    Th_Vh_disc = jax2np(compute_all_disc_avoid_terms(alg.lam, result_dt, Th_h).Th_max_lhs)
    ############################################################################################
    h_labels = task.h_labels
    figsize = np.array([5, 2.5 * task.nh])
    fig, axes = plt.subplots(task.nh, sharex=True, figsize=figsize, layout="constrained")
    for ii, ax in enumerate(axes):
        ax.plot(T_t, Th_h[:, ii], color="C2", ls="--")
        ax.plot(T_t, Th_Vh_disc[:, ii], color="C3", lw=2.0, alpha=0.5)
        ax.plot(T_t, Th_Vh[:, ii], color="C1", lw=1.0)
        ax.set_title(h_labels[ii])

    lines = [lline("C2"), lline("C3"), lline("C1")]
    axes[-1].legend(lines, ["h", "Vh_disc", "Vh pred"], loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(plot_dir / f"Vpred{cid}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)
