import os
import sys
import time

import casadi as cs
import horizon.problem as prb
import ipdb
import jax
import matplotlib.pyplot as plt
import numpy as np
from horizon.misc_function import shift_array
from horizon.solvers.solver import Solver
from horizon.transcriptions.transcriptor import Transcriptor
from horizon.utils.recedingHandler import RecedingHandler
from loguru import logger

from clfrl.dyn.step_zoh import get_step_zoh
from clfrl.dyn_cs.doubleint_cs import DoubleIntCS
from clfrl.mpc.mpc import MPCCfg, mpc_sim_single
from clfrl.solvers.snopt_casadi import SnoptSolver
from clfrl.utils.hidden_print import HiddenPrints
from clfrl.utils.jax_utils import jax_use_cpu
from clfrl.utils.paths import get_script_plot_dir


def main():
    jax_use_cpu()

    plot_dir = get_script_plot_dir()
    n_nodes = 30
    dt = 0.025

    task = DoubleIntCS()
    nom_pol = task.task.nom_pol_osc

    # x0 = np.array([-0.2, 1.4])
    # x0 = np.array([-0.95, 1.7])
    # x0 = np.array([0.73797468, 0.82405063])
    x0 = np.array([0.73797468, 0.55822785])

    cfg = MPCCfg(n_nodes, dt, cost_reg=1e-3, mpc_T=70)
    sol = mpc_sim_single(task, x0, nom_pol, cfg)
    logger.info("Took {:8.2e}s!".format(sol.t_solve))

    ###############################################################
    fig, ax = plt.subplots(layout="constrained")
    ax.plot(sol.S_x[:, 0], sol.S_x[:, 1], color="C1", lw=0.7, marker="o", ms=1.0, zorder=10)
    # ax.plot(T_x_after[:, 0], T_x_after[:, 1], color="C2", lw=0.5, marker="s", ms=0.9, zorder=9)
    task.task.plot_phase(ax)
    # for hh, T_x in enumerate(HT_x):
    #     ax.plot(T_x[:, 0], T_x[:, 1], lw=0.1, alpha=0.7, color="C3")

    fig.savefig(plot_dir / "boxshaped.pdf")
    plt.close(fig)

    ###############################################################
    x_labels, u_labels = task.task.x_labels, task.task.u_labels

    fig, axes = plt.subplots(task.nx + task.nu, sharex=True, layout="constrained")
    for ii, ax in enumerate(axes[: task.nx]):
        ax.plot(sol.S_x[:, ii], color="C1")

        for hh, T_x in enumerate(sol.ST_x):
            arange = hh + np.arange(n_nodes + 1)
            ax.plot(arange, T_x[:, ii], lw=0.3, color="C3")

        ax.set_title(x_labels[ii])
    for ii, ax in enumerate(axes[task.nx :]):
        ax.plot(sol.S_u[:, ii], color="C1")
        ax.plot(sol.S_unom, color="C3", ls="--", alpha=0.7)
        ax.set_title(u_labels[ii])
    fig.savefig(plot_dir / "boxshaped_traj.pdf")


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
