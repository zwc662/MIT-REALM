import ipdb
import jax
import matplotlib.pyplot as plt
import numpy as np

from clfrl.utils.costconstr_utils import poly4_clip_max_flat, poly4_softclip_flat
from clfrl.utils.path_utils import mkdir
from clfrl.utils.paths import get_script_plot_dir


def main():
    fns = {"poly4_softclip_flat": poly4_softclip_flat, "poly4_clip_max_flat": poly4_clip_max_flat}
    plot_dir = mkdir(get_script_plot_dir() / "costconstr")

    for fn_name, fn in fns.items():
        xs = np.linspace(-1.0, 8, 100)
        ys = fn(xs)
        dys = jax.vmap(jax.grad(poly4_softclip_flat))(xs)
        d2ys = jax.vmap(jax.grad(jax.grad(poly4_softclip_flat)))(xs)

        fig, axes = plt.subplots(3, constrained_layout=True)
        axes[0].plot(xs, ys)
        axes[1].plot(xs, dys)
        axes[2].plot(xs, d2ys)
        fig.suptitle("{}".format(fn_name))
        axes[0].set_title("f(x)")
        axes[1].set_title("f'(x)")
        axes[2].set_title("f''(x)")
        fig.savefig(plot_dir / "{}.pdf".format(fn_name))

    ######################################################
    fig, ax = plt.subplots(layout="constrained")
    xs = np.linspace(-1.0, 8, 100)
    ms = [0.1, 0.4, 0.6, 0.8, 1.0]
    for m in ms:
        ys = poly4_softclip_flat(xs, m)
        ax.plot(xs, ys, label=f"m={m:.2f}")
    ax.legend()
    fig.savefig(plot_dir / "poly4_softclip_flat_varym.pdf")

    ######################################################
    fig, ax = plt.subplots(layout="constrained")
    xs = np.linspace(-1.0, 8, 100)
    ms = [0.5, 1.0, 1.5, 2.0]
    for m in ms:
        ys = poly4_clip_max_flat(xs, m)
        ax.plot(xs, ys, label=f"max_val={m:.2f}")
    ax.legend()
    fig.savefig(plot_dir / "poly4_clip_max_flat_varym.pdf")


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
