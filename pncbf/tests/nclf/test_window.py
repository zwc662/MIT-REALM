from typing import TypeVar

import einops as ei
import ipdb
import jax.lax as lax
import jax.random as jr
import numpy as np
from flax import struct
from numpy.lib.stride_tricks import sliding_window_view
from typing_extensions import Self

from pncbf.shaper.de_shaper import DeShaper
from pncbf.shaper.deshaper_dset_buffer import DeShaperDsetBuffer
from pncbf.utils.jax_types import BoolScalar, IntScalar
from pncbf.utils.jax_utils import jax_vmap, tree_len, tree_map
from pncbf.utils.rng import PRNGKey


def main():
    T = 9
    b = 3
    T_win = 4

    T_x = np.arange(1, T + 1)
    bT_x = ei.repeat(T_x, "... -> b ...", b=b)
    bT_x = bT_x + (10 * np.arange(b)[:, None])

    # (b, T, ...) -> (b, n_windows, ..., T_win)
    bT_windows = sliding_window_view(bT_x, window_shape=T_win, axis=1)
    # print(bT_x.shape)
    # print(bT_windows.shape)

    buf = DeShaperDsetBuffer(seed=0, dset_len_max=16)
    buf._dset = DeShaper.CollectData(bT_x[:, :, None], bT_x[:, :, None], None)

    print(bT_x)

    batch = buf.sample_Vhx_batch(n_rng=30, n_zero=0, n_consec=T_win)
    print("bT_x: ", batch.bT_x.shape)
    print(batch.bT_x.squeeze(-1))


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
