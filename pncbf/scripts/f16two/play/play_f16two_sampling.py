import ipdb
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from jax_f16.f16 import F16
from lovely_histogram import plot_histogram

import run_config.int_avoid.f16two_cfg
from pncbf.dyn.f16_two import F16Two
from pncbf.dyn.sim_cts import SimCtsReal
from pncbf.dyn.sim_cts_pbar import SimCtsPbar
from pncbf.utils.jax_utils import jax_jit_np, jax_vmap, merge01, signif
from pncbf.utils.paths import get_script_plot_dir


def main():
    plot_dir = get_script_plot_dir()
    task = F16Two()

    # Sample from the initial distribution.
    n_sample = 256

    key = jr.PRNGKey(65812425)
    b_x0 = task.sample_train_x0_diffangle(key, n_sample)

    CFG = run_config.int_avoid.f16two_cfg.get(0)
    rollout_T = CFG.alg_cfg.train_cfg.rollout_T
    rollout_dt = CFG.alg_cfg.train_cfg.rollout_dt
    tf = rollout_dt * (rollout_T + 0.001)
    sim = SimCtsReal(task, task.nom_pol_pid, tf, rollout_dt, use_obs=False, max_steps=512, use_pid=False)
    bT_x, bT_t, _ = jax_jit_np(jax_vmap(sim.rollout_plot))(b_x0)

    #################################################################
    # Plot the trajectories of plane1.
    fig, ax = plt.subplots(layout="constrained")
    for ii in range(n_sample):
        ax.plot(bT_x[ii, :, task.PE1], bT_x[ii, :, task.PN1], color="C1", lw=0.2, alpha=0.8)
    ax.scatter(bT_x[:, 0, task.PE1], bT_x[:, 0, task.PN1], color="C1", s=10)
    for ii in range(n_sample):
        ax.plot(bT_x[ii, :, task.PE0], bT_x[ii, :, task.PN0], color="C3", lw=0.2, alpha=0.8)
    ax.scatter(bT_x[:, 0, task.PE0], bT_x[:, 0, task.PN0], color="C3", s=10)
    ax.set(aspect="equal")
    fig.savefig(plot_dir / "dist_sample_diffangle.pdf")
    plt.close(fig)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
