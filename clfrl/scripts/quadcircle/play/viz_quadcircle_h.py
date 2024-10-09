import ipdb
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from clfrl.dyn.quadcircle import QuadCircle
from clfrl.dyn.task import Task
from clfrl.plotting.contour_utils import centered_norm
from clfrl.plotting.plotstyle import PlotStyle
from clfrl.utils.jax_utils import jax2np, rep_vmap
from clfrl.utils.paths import get_script_plot_dir


def main():
    plot_dir = get_script_plot_dir()
    task = QuadCircle()

    # Move quad1 around.
    setup: Task.Phase2DSetup
    for ii, setup in enumerate(task.phase2d_setups()):
        logger.info(f"Plotting {setup.plot_name}...")
        bb_x, bb_Xs, bb_Ys = task.get_contour_x0(setup=ii, n_pts=128)

        bbh_h = jax2np(rep_vmap(task.h_components, rep=2)(bb_x))
        bb1_h_max = bbh_h.max(axis=2, keepdims=True)
        bbh_h = np.concatenate([bbh_h, bb1_h_max], axis=2)

        bb_issafe = jax2np(rep_vmap(task.assert_is_safe, rep=2)(bb_x))

        vmin, vmax = bbh_h.min(), bbh_h.max()
        norm = centered_norm(vmin, vmax)
        levels = np.linspace(vmin, vmax, 14)
        # Forcefully add 0 in the levels.
        levels = np.sort(np.concatenate([levels, [0.0]]))

        fig, axes = plt.subplots(2, 3, figsize=(8, 6), layout="constrained")
        axes = axes.flatten()
        for ii, ax in enumerate(axes[:5]):
            bb_h = bbh_h[:, :, ii]
            cs0 = ax.contourf(bb_Xs, bb_Ys, bb_h, norm=norm, levels=levels, cmap="RdBu_r")
            if bb_h.min() < 0 and bb_h.max() > 0:
                cs1 = ax.contour(
                    bb_Xs, bb_Ys, bb_h, levels=[0], colors=[PlotStyle.ZeroColor], alpha=0.98, linewidths=1.0
                )
            setup.plot(ax)
            if ii < task.nh:
                ax.set_title(task.h_labels[ii])
            else:
                ax.set_title("max: {:.1f}".format(bbh_h[:, :, ii].max()))
        [ax.axis("off") for ax in axes[5:]]

        cbar = fig.colorbar(cs0, ax=axes.ravel().tolist(), shrink=0.95)
        cbar.add_lines(cs1)
        fig.savefig(plot_dir / f"h_{setup.plot_name}.pdf")
        plt.close(fig)

        fig, ax = plt.subplots(layout="constrained")
        cs0 = ax.contourf(bb_Xs, bb_Ys, bb_issafe)
        cbar = fig.colorbar(cs0, ax=ax, shrink=0.95)
        setup.plot(ax)
        fig.savefig(plot_dir / f"issafe_{setup.plot_name}.pdf")
        plt.close(fig)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
