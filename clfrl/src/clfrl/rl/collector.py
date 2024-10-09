from typing import Callable, NamedTuple

import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from attrs import define

import clfrl.utils.typed_ft as ft
from clfrl.dyn.dyn_types import (
    BControl,
    BHFloat,
    BLFloat,
    BObs,
    BPolObs,
    BState,
    BTControl,
    BTHFloat,
    BTLFloat,
    BTPolObs,
    BTState,
    BTVObs,
    BVObs,
    PolObs,
    State,
    TState,
)
from clfrl.dyn.task import Task
from clfrl.utils.jax_types import BBool, BFloat, BInt, BTFloat
from clfrl.utils.jax_utils import concat_at_end, jax2np, jax_vmap, rep_vmap
from clfrl.utils.rng import PRNGKey
from clfrl.utils.shape_utils import assert_shape
from clfrl.utils.tfp import tfd


class CollectState(NamedTuple):

    b_t: BInt
    b_state: BState
    bl_lsum: BLFloat
    bh_hmax: BHFloat

    # State.
    b_status: BInt

    @staticmethod
    def HMAX_INIT():
        return -31415.9

    @staticmethod
    def create(key: PRNGKey, task: Task, collect_batch: int) -> "CollectState":
        b_t = jnp.zeros(collect_batch, dtype=jnp.int32)
        b_x0 = task.sample_train_x0(key, collect_batch)
        bl_lsum = jnp.zeros((collect_batch, task.nl), dtype=jnp.float32)
        bh_hmax = jnp.full((collect_batch, task.nh), CollectState.HMAX_INIT(), dtype=jnp.float32)
        b_status = jnp.full(collect_batch, CollectState.NORMAL(), dtype=jnp.int32)
        return CollectState(b_t, b_x0, bl_lsum, bh_hmax, b_status)

    @property
    def is_reset_maxtime(self) -> BBool:
        return self.b_status == self.RESET_MAXTIME()

    @staticmethod
    def NORMAL():
        return 1

    @staticmethod
    def RESET_MAXTIME():
        return 2

    @staticmethod
    def RESET_MEANAGE():
        return 3


class Experience(NamedTuple):
    bTp1_x: BTState
    bTp1_Vobs: BTVObs
    bTp1_polobs: BTPolObs
    bT_u: BTControl
    bTl_l: BTLFloat
    bTp1h_h: BTHFloat


_PolicyFn = Callable[[PolObs], tfd.Distribution]


def simulate(task: Task, T: int, policy: _PolicyFn, key: PRNGKey, x0: State):
    def body(state, key_):
        Vobs, polobs = task.get_obs(state)
        dist = policy(polobs)
        control, logprob = dist.experimental_sample_and_log_prob(seed=key_)
        state_new = task.step(state, control)
        return state_new, (state, Vobs, polobs, control, logprob)

    T_keys = jr.split(key, T)
    state_f, (T_state, T_Vobs, T_polobs, T_control, T_logprob) = lax.scan(body, x0, T_keys, T)
    Vobs_f, polobs_f = task.get_obs(state_f)

    Tp1_state = concat_at_end(T_state, state_f, axis=0)
    Tp1_Vobs = concat_at_end(T_Vobs, Vobs_f, axis=0)
    Tp1_polobs = concat_at_end(T_polobs, polobs_f, axis=0)

    return Tp1_state, Tp1_Vobs, Tp1_polobs, T_control, T_logprob


@define
class CollectorCfg:
    collect_batch: int
    collect_len: int
    max_age: int
    # Used to prematurely reset the distribution to prevent the replay buffer from having too concentrated of states.
    mean_age: int

    p_u_cancel: float = 0.05

    @property
    def p_reset(self):
        # ( 1 / p(reset) - 1 ) * collect_len = mean_age.
        return self.collect_len / (self.collect_len + self.mean_age)


class Collector:
    def __init__(self, task: Task, cfg: CollectorCfg):
        self.task = task
        self.cfg = cfg

    def init_collect(self, key: PRNGKey) -> CollectState:
        return CollectState.create(key, self.task, self.cfg.collect_batch)

    def collect(self, key: PRNGKey, state: CollectState, policy) -> tuple[CollectState, Experience]:
        policy: _PolicyFn
        batch_size = len(state.b_state)

        key_step, key_meanage, key_x0_sample = jr.split(key, 3)

        b_key_step = jr.split(key_step, batch_size)
        sim_fn = ft.partial(simulate, self.task, self.cfg.collect_len, policy)
        bTp1_x, bTp1_Vobs, bTp1_polobs, bT_u, bT_logprob = jax_vmap(sim_fn)(b_key_step, state.b_state)
        bT_x, b_x_f = bTp1_x[:, :-1], bTp1_x[:, -1]

        bTl_l = rep_vmap(self.task.l_components, rep=2)(bT_x)
        bl_lsum = jnp.sum(bTl_l, axis=1)

        bTp1h_h = rep_vmap(self.task.h_components, rep=2)(bTp1_x)
        bh_hmax = jnp.max(bTp1h_h[:, :-1], axis=1)

        b_isnew = state.b_t == 0
        b_t_new = state.b_t + self.cfg.collect_len

        # If hit max time, then reset ts to 0, reset ctx.
        b_maxtime = assert_shape(b_t_new >= self.cfg.max_age, batch_size)
        b_meanage = jr.bernoulli(key_meanage, self.cfg.p_reset, (batch_size,))
        b_should_reset = b_maxtime | b_meanage

        b_x0_sample = self.task.sample_train_x0(key_x0_sample, batch_size)

        b_t_new = jnp.where(b_should_reset, 0, b_t_new)
        b_x0_new = jnp.where(b_should_reset[:, None], b_x0_sample, b_x_f)

        bl_lsum_new = jnp.where(b_isnew[:, None], 0, state.bl_lsum + bl_lsum)
        assert_shape(bl_lsum_new, (batch_size, self.task.nl))

        bh_hmax_new = jnp.where(b_isnew[:, None], CollectState.HMAX_INIT(), jnp.maximum(state.bh_hmax, bh_hmax))
        assert_shape(bl_lsum_new, (batch_size, self.task.nl))

        b_status_new = jnp.where(
            b_maxtime,
            jnp.where(b_meanage, CollectState.RESET_MEANAGE(), CollectState.RESET_MAXTIME()),
            CollectState.NORMAL(),
        )
        state_new = CollectState(b_t_new, b_x0_new, bl_lsum_new, bh_hmax_new, b_status_new)

        bT_exp = Experience(bTp1_x, bTp1_Vobs, bTp1_polobs, bT_u, bTl_l, bTp1h_h)
        return state_new, bT_exp
