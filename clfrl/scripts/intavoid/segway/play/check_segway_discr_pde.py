import functools as ft
import pathlib

import ipdb
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import typer
from flax.training import orbax_utils
from loguru import logger

import run_config.int_avoid.f16gcas_cfg
import run_config.int_avoid.segway_cfg
import run_config.nclf.pend_cfg
import run_config.nclf_pol.pend_cfg
import wandb
from clfrl.dyn.f16_gcas import F16GCAS
from clfrl.dyn.segway import Segway
from clfrl.dyn.sim_cts_pbar import SimCtsPbar
from clfrl.ncbf.compute_disc_avoid import compute_all_disc_avoid_terms, cum_max_h
from clfrl.ncbf.int_avoid import IntAvoid
from clfrl.nclf.min_norm_control import bangbang_control
from clfrl.nclf.sim_nclf import SimNCLF
from clfrl.plotting.legend_helpers import lline
from clfrl.plotting.plot_task_summary import plot_task_summary
from clfrl.plotting.plot_utils import plot_boundaries
from clfrl.plotting.plotter import MPPlotter, Plotter
from clfrl.solvers.qp import OSQPStatus
from clfrl.training.ckpt_manager import get_ckpt_manager, save_create_args
from clfrl.training.run_dir import init_wandb_and_get_run_dir
from clfrl.utils.ckpt_utils import get_id_from_ckpt, get_run_path_from_ckpt, load_ckpt
from clfrl.utils.jax_utils import jax2np, jax_jit, jax_use_cpu, jax_use_double, jax_vmap, rep_vmap, tree_cat, tree_copy
from clfrl.utils.logging import set_logger_format
from clfrl.utils.paths import get_script_plot_dir


def main():
    jax_use_cpu()
    jax_use_double()

    plot_dir = get_script_plot_dir()
    set_logger_format()

    task = Segway()
    pol = task.nom_pol_lqr

    tf = 20.0
    dt = task.dt / 10.0
    n_steps = int(round(tf / dt))

    x0 = np.array([0.0, 0.5, 0.0, -3.5])

    lam = 0.58

    # Integrate nom policy.
    sim = SimCtsPbar(task, pol, n_steps, dt, use_pid=True, max_steps=n_steps, n_updates=2)
    T_x, T_t = jax2np(jax_jit(sim.rollout_plot)(x0))

    Th_h = jax2np(jax_vmap(task.h_components)(T_x))
    vterms = jax2np(compute_all_disc_avoid_terms(lam, dt, Th_h))
    Th_max_lhs = vterms.Th_max_lhs

    # Compute time derivative, see if it is zero.
    Th_lhs_dot = np.gradient(Th_max_lhs, dt, axis=0)

    # Vdot - lam * (V - h) = 0
    Th_Vdot = lam * (Th_max_lhs - Th_h)

    ##############################################################
    x_labels = task.x_labels
    fig, axes = plt.subplots(task.nx, sharex=True, layout="constrained")
    for ii, ax in enumerate(axes):
        ax.plot(T_t, T_x[:, ii], color="C1")
        ax.set_ylabel(x_labels[ii], rotation=0, ha="right")
    fig.savefig(plot_dir / "check_discr_pde_traj.pdf", bbox_inches="tight")
    plt.close(fig)
    ##############################################################
    h_labels = task.h_labels
    fig, axes = plt.subplots(task.nh, sharex=True, layout="constrained")
    for ii, ax in enumerate(axes):
        ax.plot(T_t, Th_h[:, ii], color="C3", ls="--", lw=0.8)
        ax.plot(T_t, Th_max_lhs[:, ii], color="C1", ls="--", lw=0.8)
        ax.set_ylabel(h_labels[ii], rotation=0, ha="right")
    axes[0].legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.0), fontsize="x-small")
    fig.savefig(plot_dir / "check_discr_pde_htraj.pdf", bbox_inches="tight")
    plt.close(fig)
    ##############################################################
    fig, axes = plt.subplots(task.nh, sharex=True, layout="constrained")
    for ii, ax in enumerate(axes):
        ax.plot(T_t, Th_lhs_dot[:, ii], color="C1", label="Th_lhs_dot")
        ax.plot(T_t, Th_Vdot[:, ii], ls="--", color="C2", label="Th_Vdot")
        ax.set_ylabel(h_labels[ii], rotation=0, ha="right")
    axes[0].legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.0), fontsize="x-small")
    fig.savefig(plot_dir / "check_discr_pde_Vdot.pdf", bbox_inches="tight")
    plt.close(fig)
    ##############################################################
    fig, axes = plt.subplots(task.nh, sharex=True, layout="constrained")
    for ii, ax in enumerate(axes):
        ax.plot(T_t, Th_lhs_dot[:, ii] - Th_Vdot[:, ii], color="C1")
        ax.set_ylabel(h_labels[ii], rotation=0, ha="right")
    fig.savefig(plot_dir / "check_discr_pde_resid.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
