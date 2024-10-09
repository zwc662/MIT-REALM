import einops as ei
from numpy.lib.stride_tricks import sliding_window_view
import ipdb
import numpy as np


def main():
    b = 9
    T = 7
    nx = 2
    # buf = np.arange(b * T * nx).reshape((b, T, nx))

    T_x = ei.repeat(10 * np.arange(T), "T -> T nx", nx=nx)
    T_x[:, 1] += 1

    bT_x = ei.repeat(T_x, "T nx -> b T nx", b=b)
    bT_x += 100 * (1 + np.arange(b)[:, None, None])

    print(bT_x[:, :, 0])

    n_sample = 3
    sample_len = 5

    # (b, n_windows, nx, window_len)
    bT_x_window = sliding_window_view(bT_x, window_shape=sample_len, axis=1)

    total_samples = b * (T - sample_len + 1)

    rng = np.random.default_rng(seed=5812479)
    s_idxs = rng.integers(0, total_samples, size=n_sample)

    s_idx_batch = s_idxs // (T - sample_len + 1)
    s_idx_T_i = s_idxs % (T - sample_len + 1)

    # (b, n_windows, nx, window_len) -> (n_samples, nx, window_len) -> (n_samples, window_len, nx)
    sT_x = ei.rearrange(bT_x_window[s_idx_batch, s_idx_T_i, :, :], "s ... w -> s w ...")
    assert sT_x.shape == (n_sample, sample_len, nx)

    print("-------------")
    print(s_idx_batch)
    print(s_idx_T_i)
    print("-------------")
    print(sT_x.shape)
    print(sT_x)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
