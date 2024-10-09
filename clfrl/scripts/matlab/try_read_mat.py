import pathlib

import ipdb
import jax.numpy as jnp
import matplotlib.pyplot as plt
import scipy
from matplotlib.colors import CenteredNorm

from clfrl.dyn.doubleint_wall import DoubleIntWall
from clfrl.plotting.plotstyle import PlotStyle
from clfrl.utils.jax_utils import jax2np, rep_vmap
from clfrl.utils.paths import get_data_dir, get_script_plot_dir


def main():
    plot_dir = get_script_plot_dir()
    data_dir = get_data_dir()
    mat_path = data_dir / "doubleint.mat"

    mat = scipy.io.loadmat(str(mat_path))
    h_coefs = mat["h_coefs_double"]

    def eval_h(x):
        x1, x2 = x
        x1_terms = jnp.array([x1**2, x1, 1])
        x2_terms = jnp.array([x2**2, x2, 1])
        h_terms = x1_terms[:, None] * x2_terms[None, :]
        assert h_terms.shape == h_coefs.shape
        return jnp.sum(h_terms * h_coefs)

    task = DoubleIntWall()
    bb_x, bb_Xs, bb_Ys = task.get_contour_x0(n_pts=128)

    bb_h = jax2np(rep_vmap(eval_h, rep=2)(bb_x))

    norm = CenteredNorm()

    fig, ax = plt.subplots()
    ax.contourf(bb_Xs, bb_Ys, bb_h, norm=norm, cmap="RdBu_r")
    ax.contour(bb_Xs, bb_Ys, bb_h, levels=[0], colors=[PlotStyle.ZeroColor], alpha=0.98, linewidths=1.0)
    task.plot_phase(ax)
    fig.savefig(plot_dir / "doubleintwall_sos.pdf")


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
