import ipdb
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import CenteredNorm

from clfrl.dyn.doubleint_wall import DoubleIntWall
from clfrl.plotting.plotstyle import PlotStyle
from clfrl.utils.jax_utils import rep_vmap
from clfrl.utils.paths import get_script_plot_dir


def main():
    task = DoubleIntWall()

    xlim = (-50.0, 1.0)
    ylim = (-3.0, 12.0)
    b_xs = np.linspace(*xlim, num=81)
    b_ys = np.linspace(*ylim, num=81)
    bb_Xs, bb_Ys = np.meshgrid(b_xs, b_ys)
    bb_x = np.stack([bb_Xs, bb_Ys], axis=-1)

    a0, a1, a2, a3 = [1.0, 1.0, 1.0, 1.0]
    umax = 1.0

    def get_Vs(state):
        p, v = state

        inp = jnp.array([p, v, umax])

        b0 = jnp.array([1.0, 0.0, 0.0])
        b1 = jnp.array([a0, 1.0, 0.0])
        b2 = jnp.array([a1 * a0, a1 + a0, 1.0])
        b3 = jnp.array([a2 * a1 * a0, (1 + a2) * (a0 + a1), a0 * a1 + a2])
        b4 = jnp.array([a3 * a2 * a1 * a0, (1 + a3) * (1 + a2) * (a1 + a0), a2 * a1 * a0 + a3 * a1 * a0 + a3 * a2])

        return jnp.dot(b0, inp), jnp.dot(b1, inp), jnp.dot(b2, inp), jnp.dot(b3, inp), jnp.dot(b4, inp)

    Vs = rep_vmap(get_Vs, rep=2)(bb_x)

    V_max = np.stack(Vs, axis=0).max(axis=0)

    plot_dir = get_script_plot_dir()

    figsize = (12, 2.5)
    ax: plt.Axes
    fig, axes = plt.subplots(1, len(Vs) + 1, layout="constrained", figsize=figsize)
    for ii, ax in enumerate(axes):
        if ii < len(Vs):
            ax.contourf(bb_Xs, bb_Ys, Vs[ii], norm=CenteredNorm(), cmap="RdBu_r", alpha=0.8)
            ax.contour(bb_Xs, bb_Ys, Vs[ii], levels=[0.0], colors=[PlotStyle.ZeroColor], alpha=0.98, linewidths=1.0)
            ax.set_title(rf"$b_{ii}$")
            ax.set(xlim=xlim, ylim=ylim)
        else:
            ax.contourf(bb_Xs, bb_Ys, V_max, norm=CenteredNorm(), cmap="RdBu_r", alpha=0.8)
            ax.contour(bb_Xs, bb_Ys, V_max, levels=[0.0], colors=[PlotStyle.ZeroColor], alpha=0.98, linewidths=1.0)
            ax.set_title(r"$\max_i b_i$")
            ax.set(xlim=xlim, ylim=ylim)

        # Plot the CI.
        vs = np.linspace(-8.0, 12.0, num=80)
        xs = -np.maximum(vs, 0.0) ** 2 / 2
        ax.plot(xs, vs, **PlotStyle.ci_line)

        # task.plot_phase(ax)
    fig.savefig(plot_dir / "iccbf.pdf")


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
