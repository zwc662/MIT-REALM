from typing import TypeVar

import einops as ei
import jax.lax as lax
import jax.random as jr
import numpy as np
from flax import struct
from numpy.lib.stride_tricks import sliding_window_view
from typing_extensions import Self

from clfrl.utils.jax_types import BoolScalar, IntScalar
from clfrl.utils.jax_utils import jax_vmap, tree_len, tree_map
from clfrl.utils.rng import PRNGKey

Item = TypeVar("Item")
BItem = TypeVar("BItem")


class ReplayBufferNstepNp(struct.PyTreeNode):
    data: BItem
    head: IntScalar
    size: IntScalar
    is_full: BoolScalar
    capacity: int = struct.field(pytree_node=False)
    traj_len: int = struct.field(pytree_node=False)

    @classmethod
    def create(cls, item_proto: Item, traj_len: int, capacity: int) -> "ReplayBufferNstepNp":
        data = tree_map(lambda x: np.array(ei.repeat(x, "... -> b ...", b=capacity)), item_proto)
        return ReplayBufferNstepNp(data, np.array(0), np.array(0), np.array(False), capacity, traj_len)

    def n_windows(self, n: int) -> int:
        return self.size * (self.traj_len - n + 1)

    def push(self, item: Item) -> Self:
        def push_fn(data, arr):
            data[insert_pos] = arr

        insert_pos = self.head
        tree_map(push_fn, self.data, item)
        self.head[...] = (insert_pos + 1) % self.capacity
        self.size[...] = lax.select(self.is_full, self.capacity, self.size + 1)
        self.is_full[...] = self.size == self.capacity
        return self

    def push_batch(self, b_item: BItem, batch_size: int | None = None) -> Self:
        def push_fn(data, arr):
            data[self.head : self.head + size1] = arr[:size1]
            data[:size2] = arr[size1:]

        if batch_size is None:
            batch_size = tree_len(b_item)

        # size2 is the number of items we wrap around from the start.
        size1 = min(batch_size, self.capacity - self.head)
        size2 = batch_size - size1
        tree_map(push_fn, self.data, b_item)

        self.head[...] = (self.head + batch_size) % self.capacity
        self.size[...] = self.capacity if self.is_full else min(self.size + batch_size, self.capacity)
        self.is_full[...] = self.size == self.capacity
        return self

    def get_at_index(self, idx: int | IntScalar) -> Item:
        if np.any(idx >= self.size):
            raise IndexError(f"Trying to index {idx}, size {self.size}")
        return tree_map(lambda x: x[idx], self.data)

    def get_nstep(self, idx: int | IntScalar, n: int) -> BItem:
        def index_body(bT_x):
            """bT_x: (batch, T, ...)"""
            n_extra = bT_x.shape[1] - self.traj_len
            # (b, n_windows, ..., window_len)
            bT_windows = sliding_window_view(bT_x, window_shape=n + n_extra, axis=1)

            n_windows = self.traj_len - n + 1
            s_idx_batch = idx // n_windows
            s_idx_T_i = idx % n_windows
            # (b, ..., window_len) -> (b, window_len, ...)
            return ei.rearrange(bT_windows[s_idx_batch, s_idx_T_i], "s ... w -> s w ...")

        n_windows_total = self.n_windows(n)
        if np.any(idx >= n_windows_total):
            raise IndexError(f"Trying to index {idx}, but there are only {n_windows_total} windows.")
        return tree_map(index_body, self.data)

    def uniform_sample_nstep(self, rng: np.random.Generator, batch_size: int, n: int) -> BItem:
        assert isinstance(batch_size, int), f"batch_size should be int, got {type(batch_size).__name__}"
        b_idx = rng.integers(0, self.n_windows(n), (batch_size,))
        return self.get_nstep(b_idx, n)
