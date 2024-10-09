import ipdb
import jax.numpy as jnp
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

from clfrl.dyn.odeint import tsit5_dense_pid
from clfrl.utils.jax_utils import jax2np, jax_use_cpu, rep_vmap, jax_vmap
from clfrl.utils.paths import get_script_plot_dir


def main():
    plot_dir = get_script_plot_dir()
    jax_use_cpu()

    c_theta = 1.0
    c_r = 1.0
    r_des = 1.0

    b_xs = np.linspace(-2.0, 2.0, num=128)
    b_ys = np.linspace(-2.0, 2.0, num=128)

    bb_Xs, bb_Ys = np.meshgrid(b_xs, b_ys)
    bb_state = np.stack([bb_Xs, bb_Ys], axis=-1)

    def vf(state):
        x, y = state
        r_sq = x**2 + y**2

        A_rot = np.array([[0.0, -1.0], [1.0, 0.0]])
        A = c_theta * A_rot - c_r * (r_sq - r_des**2) * np.eye(2)
        return A @ state

    bb_xdot = jax2np(rep_vmap(vf, rep=2)(bb_state))

    tf = 7.0
    dt0 = 0.1

    T_ts = np.linspace(0, tf, num=65)

    x0 = np.array([1.5, -1.0])
    T_xs_0 = jax2np(jax_vmap(tsit5_dense_pid(tf, dt0, vf, x0).evaluate)(T_ts))

    x0 = np.array([0.2, -0.1])
    T_xs_1 = jax2np(jax_vmap(tsit5_dense_pid(tf, dt0, vf, x0).evaluate)(T_ts))

    streamplot_c = matplotlib.colors.to_rgba("C3", 0.7)
    fig, ax = plt.subplots()
    ax.plot(T_xs_0[:, 0], T_xs_0[:, 1], color="C1", ls="--", linewidth=1.6, zorder=10)
    ax.plot(T_xs_1[:, 0], T_xs_1[:, 1], color="C2", ls="--", linewidth=1.45, zorder=10)
    ax.streamplot(bb_Xs, bb_Ys, bb_xdot[:, :, 0], bb_xdot[:, :, 1], color=streamplot_c, linewidth=0.5, zorder=5)
    ax.set(xlabel=r"$p_x$", ylabel=r"$p_y$")
    fig.savefig(plot_dir / "vector_field.pdf")


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
