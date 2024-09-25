import ipdb
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import CenteredNorm

from pncbf.utils.costconstr_utils import poly4_clip_max_flat, poly4_softclip_flat
from pncbf.utils.jax_utils import jax_jit_np, rep_vmap
from pncbf.utils.paths import get_script_plot_dir
from pncbf.utils.sdf_utils import sdf_capped_cone


def main():
    plot_dir = get_script_plot_dir()

    b_xs = np.linspace(-1200.0, 600.0, num=128)
    b_ys = np.linspace(-300.0, 300.0, num=128)

    bb_Xs, bb_Ys = np.meshgrid(b_xs, b_ys)
    bb_0 = np.zeros_like(bb_Xs)
    bb_x = np.stack([bb_Xs, bb_Ys, bb_0], axis=-1)

    def h_fn(x):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([-500.0, 0.0, 0.0])
        ra = 40.0
        rb = 80.0
        # Wake turbulence.
        h_wake = sdf_capped_cone(x, a, b, ra, rb)
        # Model plane as a sphere.
        h_plane = jnp.linalg.norm(x - np.array([-20.0, 0.0, 0.0]), axis=-1) - 42.0
        sdf = jnp.minimum(h_wake, h_plane)
        # Assume the ego F16 is a sphere of radius 30 ft.
        sdf = sdf - 30.0
        h = -sdf
        # Scale it down.
        h = h / 100.0
        h = poly4_clip_max_flat(h, max_val=1.0)
        h = -poly4_softclip_flat(-h, m=0.4)
        h = -poly4_softclip_flat(-h, m=0.5)
        return h

    bb_h = jax_jit_np(rep_vmap(h_fn, rep=2))(bb_x)
    print("hmin: {}, hmax: {}".format(bb_h.min(), bb_h.max()))

    fig, ax = plt.subplots(layout="constrained")
    cs0 = ax.contourf(bb_Xs, bb_Ys, bb_h, levels=20, norm=CenteredNorm(), cmap="RdBu_r")
    cs1 = ax.contour(bb_Xs, bb_Ys, bb_h, levels=[0.0], colors="k")
    cbar = fig.colorbar(cs0, ax=ax)
    cbar.add_lines(cs1)
    ax.set(aspect="equal")

    circ = plt.Circle((-20, 0), 42.0 + 30.0, facecolor="lime", edgecolor="magenta", zorder=20)
    ax.add_patch(circ)

    fig.savefig(plot_dir / "h_xy.pdf")


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
