import ipdb
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from clfrl.dyn.doubleint_wall import DoubleIntWall
from clfrl.plotting.legend_helpers import lline
from clfrl.utils.compare_ci import CIData
from clfrl.utils.logging import set_logger_format
from clfrl.utils.path_utils import mkdir
from clfrl.utils.paths import get_paper_data_dir, get_paper_plot_dir


def main():
    set_logger_format()
    data_dir = get_paper_data_dir() / "dbint"
    plot_dir = mkdir(get_paper_plot_dir() / "dbint")

    runs = {
        "PNCBF": "BMTI_50000.pkl",
        # "NCBF": "ncbf_WKAX_50000.pkl",
        "NCBF": "ncbf_EYTN_useeq_rng2_50000.pkl",
        "MPC T70": "mpc_dbint_T70_margin.pkl",
        "SOS": "sos.pkl",
    }
    colors = {"PNCBF": "C1", "NCBF": "C2", "MPC T70": "C6", "SOS": "C5"}

    task = DoubleIntWall()

    def mask(bb_Vh_):
        return np.ma.array(bb_Vh_ < 0.0, mask=bb_Vh_ > 0.0)

    figsize = np.array([3.0, 2.5])

    ax: plt.Axes
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")

    for name, fname in runs.items():
        logger.info("Loading {}...".format(fname))
        with open(data_dir / fname, "rb") as f:
            data: CIData = np.load(f, allow_pickle=True)

        bb_Vh = np.max(np.array(data.bbTh_h), axis=(2, 3))
        color = colors[name]
        ax.contour(
            data.bb_Xs, data.bb_Ys, bb_Vh, levels=[0.0], colors=color, linewidths=1.0, linestyles=["--"], zorder=11
        )
        ax.contourf(data.bb_Xs, data.bb_Ys, mask(bb_Vh), colors=color, alpha=0.2, zorder=10)

    lines = [lline(color) for color in colors.values()]
    ax.legend(lines, list(colors.keys()), loc="lower center", ncol=4, bbox_to_anchor=(0.5, 1.0), fontsize="x-small")
    # ax.legend(lines, list(colors.keys()))

    # if name == "MPC T70":
    #     bb_is_unsafe = bb_Vh > 0
    #     x = np.array([0.75, 0.55])
    #     bb_dist = np.linalg.norm(data.bbT_x[:, :, 0] - x, axis=-1)
    #     bb_dist[~bb_is_unsafe] = np.inf
    #     idx = np.unravel_index(np.argmin(bb_dist), data.bb_Xs.shape)
    #     print("idx: {}".format(idx))
    #     x0_unsafe = data.bbT_x[idx[0], idx[1], 0]
    #     print(x0_unsafe)
    #     ax.plot(x[0], x[1], marker="s", ms=1.0)
    #     ax.plot(x0_unsafe[0], x0_unsafe[1], marker="s", ms=1.0)

    task.plot_phase_paper(ax)
    fig.savefig(plot_dir / "dbint_ci.pdf", bbox_inches="tight")


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
