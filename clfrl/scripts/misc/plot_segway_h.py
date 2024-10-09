import ipdb
import matplotlib.pyplot as plt
import numpy as np

from clfrl.dyn.segway import Segway
from clfrl.utils.jax_utils import jax_use_cpu, jax_vmap
from clfrl.utils.path_utils import mkdir
from clfrl.utils.paths import get_script_plot_dir


def main():
    plot_dir = mkdir(get_script_plot_dir() / "segway")
    jax_use_cpu()

    b = 129
    task = Segway()

    b_states = np.zeros((b, 4))
    b_thetas = np.linspace(-0.7 * np.pi, 0.7 * np.pi, num=b)

    b_states_th = b_states + b_thetas[:, None]
    bh_h = jax_vmap(task.h_components)(b_states_th)

    h_labels = task.h_labels
    fig, ax = plt.subplots(layout="constrained")
    ax.plot(b_thetas, bh_h[:, 0], label=h_labels[0], zorder=4)
    ax.plot(b_thetas, bh_h[:, 1], label=h_labels[1], zorder=4)
    ax.axhline(task.h_unsafe_lb, ls="--", color="C4")
    ax.axhline(task.h_safe_ub, ls="--", color="C5")
    ax.set(xlabel=r"$\theta$")
    fig.savefig(plot_dir / "h_theta.pdf")
    plt.close(fig)

    b_p = np.linspace(-2.0 * task._p_max, 2.0 * task._p_max, num=b)
    b_states_p = b_states + b_p[:, None]
    bh_h = jax_vmap(task.h_components)(b_states_p)

    fig, ax = plt.subplots(layout="constrained")
    ax.plot(b_p, bh_h[:, 2], label=h_labels[0], zorder=4)
    ax.plot(b_p, bh_h[:, 3], label=h_labels[1], zorder=4)
    ax.axhline(task.h_unsafe_lb, ls="--", color="C4")
    ax.axhline(task.h_safe_ub, ls="--", color="C5")
    ax.set(xlabel=r"$p$")
    fig.savefig(plot_dir / "h_p.pdf")
    plt.close(fig)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
