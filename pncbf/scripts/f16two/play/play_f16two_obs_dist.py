import ipdb
import jax.random as jr
import run_config.int_avoid.f16two_cfg
import matplotlib.pyplot as plt
import numpy as np
from jax_f16.f16 import F16
from lovely_histogram import plot_histogram

from pncbf.dyn.f16_two import F16Two
from pncbf.dyn.sim_cts import SimCtsReal
from pncbf.dyn.sim_cts_pbar import SimCtsPbar
from pncbf.utils.jax_utils import jax_jit_np, jax_vmap, merge01, signif
from pncbf.utils.paths import get_script_plot_dir


def main():
    plot_dir = get_script_plot_dir()
    task = F16Two()

    # tf = 6.0
    # dt = task.dt
    # n_steps = int(round(tf / dt))
    # print("nsteps: {}".format(n_steps))

    # x0 = task.nominal_val_state()
    # x0[task.PSI0] = -np.pi / 2
    # x0[task.PSI1] = +np.pi / 2

    # Sample from the initial distribution.
    key = jr.PRNGKey(65812425)
    b_x0 = task.sample_train_x0(key, 1024)

    # sim = SimCtsPbar(task, task.nom_pol_pid, n_steps, dt, max_steps=n_steps, use_pid=False, solver="bosh3")
    # bT_x, bT_t = jax_jit_np(jax_vmap(sim.rollout_plot))(b_x0)

    CFG = run_config.int_avoid.f16two_cfg.get(0)
    rollout_T = CFG.alg_cfg.train_cfg.rollout_T
    rollout_dt = CFG.alg_cfg.train_cfg.rollout_dt
    print("rollout dt: {}, T: {} | task dt: {}".format(rollout_dt, rollout_T, task.dt))
    tf = rollout_dt * (rollout_T + 0.001)
    sim = SimCtsReal(task, task.nom_pol_pid, tf, rollout_dt, use_obs=False, max_steps=512, use_pid=False)
    bT_x, bT_t, _ = jax_jit_np(jax_vmap(sim.rollout_plot))(b_x0)

    bT_obs, _ = jax_jit_np(jax_vmap(task.get_obs, rep=2))(bT_x)

    # Histogram of Vh for each.
    bTh_h = jax_jit_np(jax_vmap(task.h_components, rep=2))(bT_x)
    bh_Vh = np.max(bTh_h, axis=1)
    # ##################################################################
    nrows = task.nh
    figsize = np.array([12, 2.2 * nrows])
    fig, axes = plt.subplots(nrows, figsize=figsize, layout="constrained")
    for ii, ax in enumerate(axes):
        plot_histogram(bh_Vh[:, ii], ax=ax)
        ax.set_ylabel("{:02}: {}".format(ii, task.h_labels[ii]), rotation=0, ha="right")
    fig.savefig(plot_dir / "dist_h_hist.pdf")
    plt.close(fig)

    # ##################################################################
    nrows = task.n_Vobs
    figsize = np.array([12, 2.2 * nrows])
    fig, axes = plt.subplots(nrows, figsize=figsize, layout="constrained")
    for ii, ax in enumerate(axes):
        plot_histogram(bT_obs[:, :, ii].flatten(), ax=ax)
        ax.set_ylabel("{:02}: {}".format(ii, task.obs_labels[ii]), rotation=0, ha="right")
    fig.savefig(plot_dir / "dist_obs_hist.pdf")
    plt.close(fig)

    ###############################################################################
    q = 0.02
    b_obs = merge01(bT_obs)
    obs_min = signif(np.quantile(b_obs, q, axis=0), 3)
    obs_max = signif(np.quantile(b_obs, 1 - q, axis=0), 3)
    print("min: ")
    print("np.{}".format(repr(obs_min)))
    print("max: ")
    print("np.{}".format(repr(obs_max)))
    ###############################################################################

    # Plot state names where there is a nan.
    for ii in range(task.nx):
        if np.any(np.isnan(bT_x[:, :, ii])):
            print("nan in {}!".format(task.x_labels[ii]))

    # Find the index where T_x is nan.
    b_hasnan = np.any(np.isnan(bT_x), axis=(1, 2))
    idx = np.argmax(b_hasnan)

    n_nan = np.sum(b_hasnan)
    print("# nan: {}".format(n_nan))

    # # Find the index where dpx is the largest in absolute value
    # dpx_idx = 20
    # b_dpx_max = np.max(np.abs(bT_obs[:, :, dpx_idx]), axis=1)
    # dpx_argmax = np.argmax(b_dpx_max)
    # print("dpx max: {}".format(np.max(bT_obs[:, :, dpx_idx])))

    T_x, T_obs = bT_x[idx], bT_obs[idx]
    T_t = bT_t[idx]

    # Plot each obs.
    nrows = task.n_Vobs
    figsize = np.array([6, 1.0 * nrows])
    fig, axes = plt.subplots(nrows, figsize=figsize, layout="constrained")
    for ii, ax in enumerate(axes):
        ax.plot(T_t, T_obs[:, ii], color="C1", lw=1.0)
        ax.set_ylabel(task.obs_labels[ii], rotation=0, ha="right")
    fig.savefig(plot_dir / "dist_obstraj.pdf")
    plt.close(fig)
    ##################################################################
    nrows = F16.NX
    figsize = np.array([6, 1.0 * nrows])
    fig, axes = plt.subplots(nrows, figsize=figsize, layout="constrained")
    for ii, ax in enumerate(axes):
        ax.plot(T_t, T_x[:, F16.NX + ii], color="C1", lw=1.0)
        ax.set_ylabel(task.x_labels[F16.NX + ii], rotation=0, ha="right")
        # Mark regions of NaN in red.
        if np.isnan(T_x[:, F16.NX + ii]).any():
            idx_first_nan = np.argmax(np.isnan(T_x[:, F16.NX + ii]))
            ax.axvspan(T_t[idx_first_nan], T_t[-1], color="r", alpha=0.5)
    task.plot_plane_constraints(axes)
    fig.savefig(plot_dir / "dist_plane1_traj.pdf")
    plt.close(fig)

    ##################################################################

    fig, ax = plt.subplots(layout="constrained")
    ax.plot(T_x[:, task.PE0], T_x[:, task.PN0], color="C1")
    ax.plot(T_x[0, task.PE0], T_x[0, task.PN0], marker="s", ms=15, color="C1")
    ax.plot(T_x[:, task.PE1], T_x[:, task.PN1], color="C4")
    ax.plot(T_x[0, task.PE1], T_x[0, task.PN1], marker="s", ms=15, color="C4")
    task.plot_pos2d(ax)
    fig.savefig(plot_dir / "dist_pos2d.pdf")
    plt.close(fig)
    ##################################################################
    fig, ax = plt.subplots(layout="constrained")
    ax.plot(T_x[:, task.PE0], T_x[:, task.H0], color="C1")
    ax.plot(T_x[0, task.PE0], T_x[0, task.H0], marker="s", ms=15, color="C1")
    ax.plot(T_x[:, task.PE1], T_x[:, task.H1], color="C4")
    ax.plot(T_x[0, task.PE1], T_x[0, task.H1], marker="s", ms=15, color="C4")
    task.plot_eastup(ax)
    fig.savefig(plot_dir / "dist_eastup.pdf")
    plt.close(fig)
    ##################################################################


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
