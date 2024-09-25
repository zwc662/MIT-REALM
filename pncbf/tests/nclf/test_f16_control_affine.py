import ipdb
import jax
import numpy as np

from pncbf.dyn.f16_gcas import F16GCAS
from pncbf.utils.jax_utils import jax_use_cpu, jax_use_double


def main():
    jax_use_cpu()
    jax_use_double()
    np.set_printoptions(precision=3, linewidth=300)

    task = F16GCAS()

    xdot_jit = jax.jit(task.xdot)

    rng = np.random.default_rng(seed=5182421)
    for ii in range(100):
        noise = 0.1 * rng.normal(size=task.NX)
        x = task.nominal_val_state() + noise
        u = rng.uniform(-0.99, 0.99, size=task.NU)

        xdot = np.array(xdot_jit(x, u))
        xdot2 = np.array(task.f(x) + task.G(x) @ u)

        # print("1: ", xdot)
        # print("2: ", xdot2)
        # print("------------------------")
        # print("dx: ", dx)
        np.testing.assert_allclose(xdot, xdot2)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
