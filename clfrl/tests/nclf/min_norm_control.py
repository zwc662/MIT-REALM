import einops as ei
import ipdb
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from clfrl.nclf.min_norm_control import min_norm_control, min_norm_control_fixed, min_norm_control_interp
from clfrl.utils.jax_utils import jax2np, jax_vmap
from clfrl.utils.paths import get_script_plot_dir


def main():
    desc_lambda = 0.1
    V = 1.0
    Vx = np.array([1.0])
    f = np.array([0.5])
    G = np.array([[1.0]])
    u_lb, u_ub = np.array([-1]), np.array([+1])

    penalty = 10.0
    relax_eps1 = 1e-2
    relax_eps2 = 5e-1

    #     0.5 * u + 0.1 <= r
    Lf_V = np.dot(Vx, f)
    LG_V = ei.einsum(Vx, G, "nx, nx nu -> nu")
    u_opt, r, info = min_norm_control(desc_lambda, u_lb, u_ub, V, Vx, f, G, penalty, relax_eps1, relax_eps2)
    print("u: {}, r: {}".format(u_opt, r))
    x = np.array([*u_opt, r])

    u2 = -(Lf_V + desc_lambda * V) / LG_V
    r2 = 0.0
    x2 = np.array([*u2, r2])

    Q, c, qpG, h = info

    lhs = qpG @ x2
    rhs = h
    print(lhs, rhs, lhs <= rhs)

    lhs = qpG @ x
    rhs = h
    print(lhs, rhs, lhs <= rhs)

    def obj_fn(u):
        return 0.5 * u**2

    def constr_fn(u):
        constr = Lf_V + jnp.dot(LG_V, u) + desc_lambda * V
        Vdot = Lf_V + jnp.dot(LG_V, u)
        return constr, Vdot

    b_us = np.linspace(-1.2, 1.2, num=96)
    b_obj = jax2np(jax_vmap(obj_fn)(b_us))
    b_constr, b_Vdot = jax2np(jax_vmap(constr_fn)(b_us))

    fig, ax = plt.subplots(layout="constrained")
    ax.plot(b_us, b_obj, color="C1")
    ax.plot(b_us, b_constr, color="C0")
    ax.plot(b_us, b_constr - r, color="C2", ls="--")
    ax.plot(b_us, b_Vdot, color="C3")
    ax.axvline(u_opt, color="C4")

    for ii, interp in enumerate([0.2, 0.4, 0.6, 0.8]):
        u_opt, _, _ = min_norm_control_interp(
            desc_lambda, interp, u_lb, u_ub, V, Vx, f, G, penalty, relax_eps1, relax_eps2
        )
        print("{:.2e} - {}".format(interp, u_opt))
        ax.axvline(u_opt, color=f"C{ii}", label="int{}".format(interp))

    for ii, rate in enumerate([0.1, 0.2, 0.3, 0.4, 0.5]):
        u_opt, _, _ = min_norm_control_fixed(rate, u_lb, u_ub, Vx, f, G, penalty, relax_eps1, relax_eps2)
        print("{:.2e} - {}".format(rate, u_opt))
        ax.axvline(u_opt, color=f"C{ii}", label="rate{}".format(rate), ls="--")

    ax.legend()

    plot_dir = get_script_plot_dir()
    fig.savefig(plot_dir / "min_norm_control_test0.pdf")


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
