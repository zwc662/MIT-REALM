import einops as ei
import ipdb
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from pncbf.ncbf.min_norm_cbf import min_norm_cbf
from pncbf.utils.jax_utils import jax2np, jax_vmap
from pncbf.utils.paths import get_script_plot_dir


def main():
    alpha = 0.1
    V = np.array(1.0)
    Vx = np.array([1.0])
    f = np.array([0.5])
    # f = np.array([1.2])
    G = np.array([[1.0]])
    u_lb, u_ub = np.array([-1]), np.array([+1])

    u_nom = np.array([0.0])

    h_V, h_Vx = V[None], Vx[None]

    penalty = 10.0
    relax_eps1 = 1e-2
    relax_eps2 = 5e-1

    #    LfV + LGV u - lam * V <= r
    #     0.5 + u - 0.1 <= r
    #           u <= -0.4 + r
    Lf_V = np.dot(Vx, f)
    LG_V = ei.einsum(Vx, G, "nx, nx nu -> nu")
    u_opt, r, info = min_norm_cbf(alpha, u_lb, u_ub, h_V, h_Vx, f, G, u_nom, penalty, relax_eps1, relax_eps2)
    print("u: {}, r: {}".format(u_opt, r))
    x = np.array([*u_opt, r])

    def obj_fn(u):
        return 0.5 * (u - u_nom) ** 2

    def constr_fn(u):
        constr = Lf_V + jnp.dot(LG_V, u) - alpha * V
        Vdot = Lf_V + jnp.dot(LG_V, u)
        return constr, Vdot

    b_us = np.linspace(-1.2, 1.2, num=96)
    b_obj = jax2np(jax_vmap(obj_fn)(b_us))
    b_constr, b_Vdot = jax2np(jax_vmap(constr_fn)(b_us))

    fig, ax = plt.subplots(layout="constrained")
    ax.plot(b_us, b_obj, color="C1", label="Obj")
    ax.plot(b_us, b_constr, color="C0", label="Constr")
    ax.plot(b_us, b_constr - r, color="C2", ls="--", label="Relaxed Constr")
    ax.plot(b_us, b_Vdot, color="C3", label=r"$\dot{V}$")
    ax.axvline(u_opt, color="C4", label=r"$u_{cbf}$")
    ax.axvline(u_nom, color="C5", label=r"$u_{nom}$")
    ax.legend()

    plot_dir = get_script_plot_dir()
    fig.savefig(plot_dir / "min_norm_cbf_test0.pdf")


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
