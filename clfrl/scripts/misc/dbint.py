import ipdb
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import CenteredNorm

from clfrl.dyn.doubleint_wall import DoubleIntWall
from clfrl.utils.paths import get_script_plot_dir


def main():
    task = DoubleIntWall()
    plot_dir = get_script_plot_dir()

    b_xs = np.linspace(-2, 2, num=128)
    b_vs = np.linspace(-2, 2, num=128)
    bb_Xs, bb_Ys = np.meshgrid(b_xs, b_vs)
    # bb_x = np.stack([bb_Xs, bb_Ys], axis=-1)

    bb_V = bb_Xs + 0.5 * np.maximum(0, bb_Ys) ** 2 - task.pos_wall

    fig, ax = plt.subplots(layout="constrained")
    task.plot_phase(ax)
    ax.contourf(bb_Xs, bb_Ys, bb_V, levels=31, norm=CenteredNorm(), cmap="RdBu_r")
    fig.savefig(plot_dir / "dbint_ttc.pdf")


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
