import ipdb
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from jax_f16.f16 import F16
from lovely_histogram import plot_histogram

import run_config.int_avoid.f16two_cfg
from pncbf.dyn.f16_gcas import compute_f16_vel_angles
from pncbf.dyn.f16_two import F16Two
from pncbf.dyn.sim_cts import SimCtsReal
from pncbf.dyn.sim_cts_pbar import SimCtsPbar
from pncbf.utils.jax_utils import jax_jit_np, jax_vmap, merge01, signif
from pncbf.utils.path_utils import mkdir
from pncbf.utils.paths import get_script_plot_dir


def main():
    plot_dir = mkdir(get_script_plot_dir() / "check_flip")
    task = F16Two()

    n_sample = 256

    key = jr.PRNGKey(65812425)
    sample_fns = {
        "rng": task.sample_train_x0_random,
        "infront": task.sample_train_x0_infront,
        "point": task.sample_train_x0_point,
        "diffangle": task.sample_train_x0_diffangle,
    }
    b_x0s = {k: sample_fn(key, n_sample) for k, sample_fn in sample_fns.items()}

    CFG = run_config.int_avoid.f16two_cfg.get(0)
    rollout_T = CFG.alg_cfg.train_cfg.rollout_T
    rollout_dt = CFG.alg_cfg.train_cfg.rollout_dt
    tf = rollout_dt * (rollout_T + 0.001)
    sim = SimCtsReal(task, task.nom_pol_pid, tf, rollout_dt, use_obs=False, max_steps=rollout_T + 2, use_pid=False)
    rollout_fn = jax_jit_np(jax_vmap(sim.rollout_plot))
    results = {k: rollout_fn(b_x0) for k, b_x0 in b_x0s.items()}
    bT_xs = {k: bT_x for k, (bT_x, _, _) in results.items()}
    bT_t = results["rng"][1]

    # Make sure they are all finite.
    for bT_x in bT_xs.values():
        if not np.isfinite(bT_x).all():
            raise ValueError("bT_x is not finite.")

    ###################################################
    # Plot the distribution of collision.
    fig, axes = plt.subplots(len(bT_xs), layout="constrained", sharex=True)
    for ii, (key, bT_x) in enumerate(bT_xs.items()):
        bTh_h = jax_jit_np(jax_vmap(task.h_components, rep=2))(bT_x)
        b_hcol = bTh_h[:, :, -1].max(-1)

        plot_histogram(b_hcol, ax=axes[ii])
        axes[ii].set_ylabel(key)
    fig.savefig(plot_dir / "h_col_hist.pdf")
    plt.close(fig)

    # #################################################################################
    # for idx in list(range(n_sample))[:16]:
    #     T_x = bT_x[idx]
    #     # T_x_flip = bT_x_flip[idx]
    #
    #     fig, ax = plt.subplots(layout="constrained")
    #     ax.plot(T_x[:, task.PE0], T_x[:, task.PN0], color="C1", zorder=8)
    #     ax.plot(T_x[0, task.PE0], T_x[0, task.PN0], marker="s", ms=10, color="C1")
    #
    #     t_point = 2.0
    #     x0, x1 = T_x_flip[0, : F16.NX], T_x_flip[0, F16.NX :]
    #     # [north, east]
    #     vel_angles = compute_f16_vel_angles(x1)
    #     x1_point = x1[F16.POS_NEU] + t_point * x1[F16.VT] * vel_angles
    #     ax.plot(x1_point[1], x1_point[0], marker="^", color="C0", zorder=10)
    #     pos1mpos0 = x1_point[:2] - x0[F16.POS2D_NED]
    #
    #     yaw_point = np.arctan2(pos1mpos0[1], pos1mpos0[0])
    #     # Visualize the yaw by drawing a line.
    #     p0 = x0[F16.POS2D_NED]
    #     p1 = p0 + 5_000 * np.array([np.cos(yaw_point), np.sin(yaw_point)])
    #     # (T, 2)
    #     line = np.stack([p0, p1], axis=0)
    #     # The line is [north ,east]. We want to plot [east, north].
    #     ax.plot(line[:, 1], line[:, 0], color="C0", ls="-.", zorder=12)
    #
    #     ax.plot(T_x_flip[:, task.PE0], T_x_flip[:, task.PN0], ls="--", color="C2", zorder=8)
    #     ax.plot(T_x_flip[0, task.PE0], T_x_flip[0, task.PN0], marker="s", ms=10, color="C1")
    #
    #     ax.plot(T_x[:, task.PE1], T_x[:, task.PN1], color="C4")
    #     ax.plot(T_x[0, task.PE1], T_x[0, task.PN1], marker="s", ms=10, color="C4")
    #     task.plot_pos2d(ax)
    #     fig.suptitle("h={}, hflip={}".format(b_hcol[idx], b_hcol_flip[idx]))
    #     fig.savefig(plot_dir / "{:2}_dist_pos2d.pdf".format(idx))
    #     plt.close(fig)
    #
    #     ##################################################################
    #     fig, ax = plt.subplots(layout="constrained")
    #     ax.plot(T_x[:, task.PE0], T_x[:, task.H0], color="C1", zorder=8)
    #     ax.plot(T_x[0, task.PE0], T_x[0, task.H0], marker="s", ms=10, color="C1")
    #
    #     ax.plot(T_x_flip[:, task.PE0], T_x_flip[:, task.H0], ls="--", color="C2", zorder=8)
    #     ax.plot(T_x_flip[0, task.PE0], T_x_flip[0, task.H0], marker="s", ms=10, color="C1")
    #
    #     ax.plot(T_x[:, task.PE1], T_x[:, task.H1], color="C4")
    #     ax.plot(T_x[0, task.PE1], T_x[0, task.H1], marker="s", ms=10, color="C4")
    #     task.plot_eastup(ax)
    #     fig.suptitle("h={}, hflip={}".format(b_hcol[idx], b_hcol_flip[idx]))
    #     fig.savefig(plot_dir / "{:2}_dist_eastup.pdf".format(idx))
    #     plt.close(fig)
    #     ##################################################################


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
