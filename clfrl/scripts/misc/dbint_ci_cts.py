import ipdb
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.colors import Normalize

from clfrl.dyn.doubleint_wall import DoubleIntWall
from clfrl.dyn.sim_cts import SimCtsReal
from clfrl.plotting.contour_utils import centered_norm
from clfrl.utils.jax_utils import jax2np, jax_jit, jax_vmap, rep_vmap
from clfrl.utils.logging import set_logger_format
from clfrl.utils.path_utils import mkdir
from clfrl.utils.paths import get_script_plot_dir


def main():
    set_logger_format()
    plot_dir = mkdir(get_script_plot_dir() / "doubleint" / "cts")
    task = DoubleIntWall()

    nom_pol = task.nom_pol_osc
    x0 = np.array([-0.6, 1.5])
    tf = 80.0 * task.dt
    logger.info("Integrating for tf={:.1f}".format(tf))

    for ii, setup in enumerate(task.phase2d_setups()):
        logger.info("Plotting {}...".format(setup.plot_name))

        bb_x, bb_Xs, bb_Ys = task.get_contour_x0(setup=ii, n_pts=256)

        def get_Th(state0):
            sim = SimCtsReal(task, nom_pol, tf, result_dt=task.dt)
            T_x, T_t, stats = sim.rollout_plot(state0)
            Th_h = jax_vmap(task.h_components)(T_x)
            return Th_h, T_x, T_t, stats

        _, T_x, T_t, _ = jax_jit(get_Th)(x0)

        logger.info("Rollout...")
        bbTh_h, bbT_x, _, bb_stats = jax2np(jax_jit(rep_vmap(get_Th, rep=2))(bb_x))
        logger.info("Max steps: {} / {}".format(bb_stats["num_steps"].max(), bb_stats["max_steps"].min()))

        # Plot how many time before constraint violation.
        def get_t_violate(T_h):
            assert T_h.ndim == 1
            has_violate = jnp.any(T_h > 0.0)
            idx_violate = jnp.argmax(T_h >= 0.0)
            return jnp.where(has_violate, T_t[idx_violate], jnp.inf)

        bbh_t_viol = jax2np(jax_jit(rep_vmap(jax_vmap(get_t_violate, in_axes=1), rep=2))(bbTh_h))
        bb1_t_viol_min = np.min(bbh_t_viol, axis=2, keepdims=True)
        bbh_t_viol = np.concatenate([bbh_t_viol, bb1_t_viol_min], axis=2)
        bbh_t_viol = np.ma.array(bbh_t_viol, mask=~np.isfinite(bbh_t_viol))

        t_viol_min = bb1_t_viol_min.flatten()
        t_viol_min = t_viol_min[np.isfinite(t_viol_min)]

        norm = Normalize(vmin=0, vmax=1.005 * t_viol_min.max())

        fig, ax = plt.subplots(layout="constrained")
        ax.hist(t_viol_min, bins=48)
        fig.savefig(plot_dir / f"time_to_violate_dist_{setup.plot_name}.pdf")
        plt.close(fig)

        figsize = (2 * (task.nh + 1), 2)
        fig, axes = plt.subplots(1, task.nh + 1, figsize=figsize, layout="constrained")
        for ii, ax in enumerate(axes):
            cs0 = ax.contourf(bb_Xs, bb_Ys, bbh_t_viol[:, :, ii], norm=norm, levels=15, cmap="rocket_r")
            setup.plot(ax)
            if ii < task.nh:
                ax.set_title(task.h_labels[ii])
            else:
                ax.set_title("max: {:.1f}".format(bbh_t_viol[:, :, ii].max()))

        fig.colorbar(cs0, ax=axes.ravel().tolist(), shrink=0.95)
        fig.savefig(plot_dir / f"time_to_violate_{setup.plot_name}.pdf")
        plt.close(fig)

        ###########################################################################
        bbh_Vh = bbTh_h.max(axis=2)
        bb1_Vh_max = bbh_Vh.max(axis=2, keepdims=True)
        bbh_Vh = np.concatenate([bbh_Vh, bb1_Vh_max], axis=2)

        h_labels = [*task.h_labels, "Max"]

        vmin, vmax = bbh_Vh.min(), bbh_Vh.max()
        norm = centered_norm(vmin, vmax)

        figsize = (2 * (task.nh + 1), 2)
        fig, axes = plt.subplots(1, task.nh + 1, figsize=figsize, layout="constrained")
        for jj, ax in enumerate(axes):
            ax.plot(T_x[:, setup.idx0], T_x[:, setup.idx1], "C1", linewidth=1.0, zorder=10)
            setup.plot(ax)
            ax.contourf(bb_Xs, bb_Ys, bbh_Vh[:, :, jj], levels=11, norm=norm, cmap="RdBu_r", zorder=3.5, alpha=0.9)
            ax.contour(bb_Xs, bb_Ys, bbh_Vh[:, :, jj], levels=[0.0], colors=["C5"], zorder=10.0)
            ax.set_title(h_labels[jj])
        fig.savefig(plot_dir / f"ci_cts_{setup.plot_name}.pdf")

    # logger.info("Plotting...")
    # x_labels = task.x_labels
    # fig, axes = plt.subplots(task.nx, layout="constrained")
    # for ii, ax in enumerate(axes):
    #     ax.plot(T_t, T_x[:, ii], marker="o", ms=1.5, color="C1")
    #     ax.set_title(x_labels[ii])
    # plot_boundaries(axes, task.train_bounds())
    # fig.savefig(plot_dir / "traj_cts.pdf")
    # plt.close(fig)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
