import jax.lax as lax
import jax.numpy as jnp
import numpy as np

from pncbf.utils.jax_utils import jax_use_cpu


def test_cummax():
    jax_use_cpu()

    T_h = np.array([1.0, 0.5, 0.4, 0.2, 1.2, 0.3, 1.3, 0.1])
    T = len(T_h)
    T_cummax_true = np.array([T_h[:ii].max() for ii in range(1, T + 1)])

    Th_h = np.stack([T_h, 0.5 * T_h + 1.0], axis=1)
    Th_cummax_true = np.stack([T_cummax_true, 0.5 * T_cummax_true + 1.0], axis=1)

    T_cummax = np.array(lax.cummax(T_h, axis=0))
    assert T_cummax.shape == (T,)
    np.testing.assert_allclose(T_cummax, T_cummax_true)

    # (T, 2)
    Th_cummax = np.array(lax.cummax(Th_h, axis=0))
    assert Th_cummax.shape == (T, 2)
    np.testing.assert_allclose(Th_cummax, Th_cummax_true)


def main():
    test_cummax()


if __name__ == "__main__":
    main()
