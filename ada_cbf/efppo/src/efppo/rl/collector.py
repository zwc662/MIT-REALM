import functools as ft
from typing import Any, NamedTuple, Tuple

import numpy as np

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from attrs import define
from flax import struct

from efppo.task.dyn_types import TControl, THFloat, TObs
from efppo.task.task import Task, TaskState
from efppo.utils.cfg_utils import Cfg
from efppo.utils.jax_types import FloatScalar, IntScalar, TFloat
from efppo.utils.jax_utils import concat_at_front, concat_at_end
from efppo.utils.rng import PRNGKey
from efppo.utils.tfp import tfd


@define
class CollectorCfg(Cfg):
    # Batch size when collecting data.
    n_envs: int
    # How long to rollout before updating.
    rollout_T: int
    # The mean age of the data before resetting.
    mean_age: int
    # If the age is greater than this, then reset.
    max_T: int


class CollectorState(NamedTuple):
    steps: IntScalar
    state: TaskState
    z: FloatScalar


class RolloutOutput(NamedTuple):
    Tp1_state: TaskState
    Tp1_obs: TObs
    Tp1_z: TFloat
    T_control: TControl
    T_logprob: TFloat
    T_l: TFloat
    Th_h: THFloat


def collect_single_mode(
    task: Task,
    x0: TaskState,
    z0: float,
    get_pol,
    disc_gamma: float,
    z_min: float,
    z_max: float,
    rollout_T: int,
):
    def _body(state: CollectorState, _):
        obs_pol = task.get_obs(state.state)
        a_pol: tfd.Distribution = get_pol(obs_pol, state.z)
        control = a_pol.mode()
        envstate_new = task.step(state.state, control)

        # Z dynamics.
        l = task.l(envstate_new, control)
        z_new = (state.z - l) / disc_gamma
        z_new = jnp.clip(z_new, z_min, z_max)
        if hasattr(task, 'l1_control_info'):
            task.l1_control_info['z'] = z_new

        return CollectorState(state.steps, envstate_new, z_new), (envstate_new, obs_pol, z_new, control)
    if hasattr(task, 'l1_control_info'):
        task.l1_control_info['z'] = z0
    colstate0 = CollectorState(0, x0, z0)
    collect_state, (T_envstate, T_obs, T_z, T_u) = lax.scan(_body, colstate0, None, length=rollout_T)
    obs_final = task.get_obs(collect_state.state)

    # Add the initial observations.
    Tp1_state = jtu.tree_map(concat_at_front, colstate0.state, T_envstate)
    T_state_fr, T_state_to = Tp1_state[:-1], Tp1_state[1:]
    Tp1_obs = jtu.tree_map(concat_at_end, T_obs, obs_final)
    Tp1_z = concat_at_front(z0, T_z)
    T_l = jax.vmap(task.l)(T_state_to, T_u)
    Th_h = jax.vmap(task.h_components)(T_state_to)

    return RolloutOutput(Tp1_state, Tp1_obs, Tp1_z, T_u, None, T_l, Th_h)

def collect_single(
    task: Task,
    key0: PRNGKey,
    colstate0: CollectorState,
    get_pol,
    disc_gamma: float,
    z_min: float,
    z_max: float,
    rollout_T: int,
):
    def _body(state: CollectorState, key):
        obs_pol = task.get_obs(state.state)
        a_pol: tfd.Distribution = get_pol(obs_pol, state.z)
        control, logprob = a_pol.experimental_sample_and_log_prob(seed=key)
        envstate_new = task.step(state.state, control)
        
        # Z dynamics.
        l = task.l(envstate_new, control)
        z_new = (state.z - l) / disc_gamma
        z_new = jnp.clip(z_new, z_min, z_max)

        return CollectorState(state.steps, envstate_new, z_new), (envstate_new, obs_pol, z_new, control, logprob)

    # Randomly sample z0.
    key_z0, key_step = jr.split(key0)
    z0 = jr.uniform(key_z0, minval=z_min, maxval=z_max)
    colstate0 = colstate0._replace(z=z0)

    assert colstate0.steps.shape == tuple()
    T_keys = jr.split(key_step, rollout_T)
    collect_state, (T_envstate, T_obs, T_z, T_u, T_logprob) = lax.scan(_body, colstate0, T_keys, length=rollout_T)
    collect_state = collect_state._replace(steps=collect_state.steps + rollout_T)
    obs_final = task.get_obs(collect_state.state)

    # Add the initial observations.
    Tp1_state = jtu.tree_map(concat_at_front, colstate0.state, T_envstate)
    T_state_fr, T_state_to = Tp1_state[:-1], Tp1_state[1:]
    Tp1_obs = jtu.tree_map(concat_at_end, T_obs, obs_final)
    Tp1_z = concat_at_front(z0, T_z)
    T_l = jax.vmap(task.l)(T_state_to, T_u)
    Th_h = jax.vmap(task.h_components)(T_state_to)

    return collect_state, RolloutOutput(Tp1_state, Tp1_obs, Tp1_z, T_u, T_logprob, T_l, Th_h)

def collect_single_env_mode(
    task: Task,
    x0: TaskState,
    z0: float,
    get_pol,
    disc_gamma: float,
    z_min: float,
    z_max: float,
    rollout_T: int,
):
    def _body(state: CollectorState, _):
        obs_pol = task.get_obs(state.state)
        a_pol: tfd.Distribution = get_pol(obs_pol, state.z)
        control = a_pol.mode()
        envstate_new = task.step(state.state, control)

        # Z dynamics.
        l = task.l(envstate_new, control)
        z_new = (state.z - l) / disc_gamma
        z_new = jnp.clip(z_new, z_min, z_max)
        new_state = CollectorState(state.steps, envstate_new.reshape(-1), z_new)
        return new_state, (envstate_new, obs_pol, z_new, control)
 
    colstate0 = CollectorState(0, x0, z0)
    
    T_envstate = []
    T_obs = []
    T_z = []
    T_u = [] 

    for t in range(rollout_T):
        collect_state, (envstate_new, obs_pol, z_new, control) = _body(colstate0, None)
        assert not jnp.isnan(envstate_new).any()
        assert not jnp.isnan(obs_pol).any()
        assert not jnp.isnan(control).any() 
        
        T_envstate.append(envstate_new)
        T_obs.append(obs_pol)
        T_z.append(z_new)
        T_u.append(control) 
    obs_final = task.get_obs(collect_state.state)

    # Add the initial observations.
    Tp1_state = jnp.stack((colstate0.state, *T_envstate))
    T_state_fr, T_state_to = Tp1_state[:-1], Tp1_state[1:]
    Tp1_obs = jnp.stack((*T_obs, obs_final)) 
    Tp1_z = jnp.stack((jnp.asarray([z0]), *T_z)).reshape(-1)
    T_l = []
    Th_h = []
    for state_to, u in zip(T_state_to, T_u):
        T_l.append(task.l(state_to, T_u))
        Th_h.append(task.h_components(state_to))
    
    T_l = jnp.stack(T_l)
    Th_h = jnp.stack(Th_h)

    return RolloutOutput(Tp1_state, Tp1_obs, Tp1_z, T_u, None, T_l, Th_h)

def collect_single_env(
    env_idx: int,
    task: Task,
    key0: PRNGKey,
    colstate0: CollectorState,
    get_pol,
    disc_gamma: float,
    z_min: float,
    z_max: float,
    rollout_T: int,
):
    def _body(state: CollectorState, key):
        obs_pol = task.get_obs(state.state)
        a_pol: tfd.Distribution = get_pol(obs_pol, state.z.squeeze())
        control, logprob = a_pol.experimental_sample_and_log_prob(seed=key)
        envstate_new = jnp.asarray(task.step(state.state, np.asarray(control).reshape(2)))

        # Z dynamics.
        l = task.l(envstate_new, control)
        z_new = (state.z - l) / disc_gamma
        z_new = jnp.clip(z_new, z_min, z_max)

        new_state = CollectorState(state.steps, envstate_new.reshape(-1), z_new)
        return new_state, (envstate_new, obs_pol, z_new, control, logprob)

    # Randomly sample z0.
    key_z0, key_step = jr.split(key0)
    z0 = jr.uniform(key_z0, minval=z_min, maxval=z_max)
    colstate0 = colstate0._replace(z=z0)

    T_keys = jr.split(key_step, rollout_T)

    # Initialize outputs for the loop
    T_envstate = []
    T_obs = []
    T_z = []
    T_u = []
    T_logprob = []
    collect_state = colstate0

    # Perform the loop over rollout_T
    for t in range(rollout_T):
        collect_state, (envstate_new, obs_pol, z_new, control, logprob) = _body(collect_state, T_keys[t])
        assert not jnp.isnan(envstate_new).any()
        assert not jnp.isnan(obs_pol).any()
        assert not jnp.isnan(control).any()
        assert not jnp.isnan(logprob).any()
        
        T_envstate.append(envstate_new)
        T_obs.append(obs_pol)
        T_z.append(z_new)
        T_u.append(control)
        T_logprob.append(logprob)

    # Convert lists to jnp arrays
    T_envstate = jnp.stack(T_envstate)
    T_obs = jnp.stack(T_obs)
    T_z = jnp.stack(T_z)
    T_u = jnp.stack(T_u)
    T_logprob = jnp.stack(T_logprob)
    collect_state = collect_state._replace(steps=collect_state.steps + rollout_T)

    obs_final = task.get_obs(collect_state.state)

    # Add the initial observations.
    Tp1_state = jnp.stack((colstate0.state, *T_envstate))
    T_state_fr, T_state_to = Tp1_state[:-1], Tp1_state[1:]
    Tp1_obs = jnp.stack((*T_obs, obs_final)) 
    Tp1_z = jnp.stack((jnp.asarray([z0]), *T_z)).reshape(-1)
    T_l = []
    Th_h = []
    for state_to, u in zip(T_state_to, T_u):
        T_l.append(task.l(state_to, T_u))
        Th_h.append(task.h_components(state_to))
    
    T_l = jnp.stack(T_l)
    Th_h = jnp.stack(Th_h)

    return collect_state, RolloutOutput(Tp1_state, Tp1_obs, Tp1_z, T_u, T_logprob, T_l, Th_h)

class Collector(struct.PyTreeNode):
    collect_idx: int
    key: PRNGKey
    collect_state: CollectorState
    task: Task = struct.field(pytree_node=False)
    cfg: CollectorCfg = struct.field(pytree_node=False)

    Rollout = RolloutOutput

    @classmethod
    def create(cls, key: PRNGKey, task: Task, cfg: CollectorCfg):
        key, key_init = jr.split(key)
        b_state = jnp.asarray(task.sample_x0_train(key_init, cfg.n_envs))
        b_steps = jnp.zeros(cfg.n_envs, dtype=jnp.int32)
        b_z0 = jnp.zeros(cfg.n_envs)
        collector_state = CollectorState(b_steps, b_state, b_z0)
        return Collector(0, key, collector_state, task, cfg)

    def _collect_single(
        self, key0: PRNGKey, colstate0: CollectorState, get_pol, disc_gamma: float, z_min: float, z_max: float
    ) -> tuple[CollectorState, RolloutOutput]:
        return collect_single(self.task, key0, colstate0, get_pol, disc_gamma, z_min, z_max, self.cfg.rollout_T)

    @property
    def p_reset(self) -> FloatScalar:
        return self.cfg.rollout_T / self.cfg.mean_age

    def collect_batch(
        self, get_pol, disc_gamma: float, z_min: float, z_max: float
    ) -> tuple["Collector", RolloutOutput]:
        key0 = jr.fold_in(self.key, self.collect_idx)
        key_pol, key_reset_bernoulli, key_reset = jr.split(key0, 3)
        b_keys = jr.split(key_pol, self.cfg.n_envs)
        collect_fn = ft.partial(self._collect_single, get_pol=get_pol, disc_gamma=disc_gamma, z_min=z_min, z_max=z_max)
        collect_state, bT_outputs = jax.vmap(collect_fn)(b_keys, self.collect_state)
        assert collect_state.steps.shape == (self.cfg.n_envs,)

        # Resample x0.
        b_shouldreset = jr.bernoulli(key_reset_bernoulli, self.p_reset, shape=(self.cfg.n_envs,))
        # Also reset if we exceed the max rollout length.
        b_shouldreset = jnp.logical_or(b_shouldreset, collect_state.steps >= self.cfg.max_T)
        # Also reset if the state is bad.
        b_shouldreset = b_shouldreset | jax.vmap(self.task.should_reset)(collect_state.state)
        b_state_reset = self.task.sample_x0_train(key_reset, self.cfg.n_envs)

        def reset_fn(should_reset, state_reset_new, state_reset_old, steps_old):
            def reset_fn_(arr_new, arr_old):
                return jnp.where(should_reset, arr_new, arr_old)

            state_new = jtu.tree_map(reset_fn_, state_reset_new, state_reset_old)
            steps_new = jnp.where(should_reset, 0, steps_old)
            return steps_new, state_new

        b_steps_new, b_state_new = jax.vmap(reset_fn)(
            b_shouldreset, b_state_reset, collect_state.state, collect_state.steps
        )
        collect_state = CollectorState(b_steps_new, b_state_new, collect_state.z)

        new_self = self.replace(collect_idx=self.collect_idx + 1, collect_state=collect_state)
        return new_self, bT_outputs

    def _collect_single_env(
        self, env_idx: int, key0: PRNGKey, colstate0: CollectorState, get_pol, disc_gamma: float, z_min: float, z_max: float
    ) -> tuple[CollectorState, RolloutOutput]:
        return collect_single_env(env_idx, self.task, key0, colstate0, get_pol, disc_gamma, z_min, z_max, self.cfg.rollout_T)


    def collect_batch_iteratively(
        self, get_pol, disc_gamma: float, z_min: float, z_max: float
    ) -> tuple["Collector", RolloutOutput]:
        key0 = jr.fold_in(self.key, self.collect_idx)
        key_pol, key_reset_bernoulli, key_reset = jr.split(key0, 3)
        b_keys = jr.split(key_pol, self.cfg.n_envs)
        
        # Initialize lists to accumulate results
        bT_outputs = []
        # Use a for loop instead of jax.vmap
 
        for i in range(self.cfg.n_envs):
            collect_state_i = jtu.tree_map(lambda array: array[i], self.collect_state)
            collect_state_i, bT_output = self._collect_single_env(
                i, 
                b_keys[i], 
                collect_state_i,
                get_pol=get_pol, 
                disc_gamma=disc_gamma, 
                z_min=z_min, 
                z_max=z_max
                )
       
            # Resample x0
            shouldreset_i = jr.bernoulli(key_reset_bernoulli, self.p_reset)
            shouldreset_i = jnp.logical_or(shouldreset_i, self.collect_state_i.steps >= self.cfg.max_T).item()
            shouldreset_i = shouldreset_i | self.task.should_reset(collect_state_i.state)
            if shouldreset_i:
                collect_state_i.state = self.task.sample_x0_train(key_reset, 1)[0]
                collect_state_i.steps *= 0

            bT_outputs.append(bT_output)
            self.collect_state.state.at[i].set(collect_state_i.state)
            self.collect_state.steps.at[i].set(collect_state_i.steps.item())
            self.collect_state.z.at[i].set(collect_state_i.z.item())

        assert self.collect_state.steps.shape == (self.cfg.n_envs,)
        # Stack the collected states and outputs
        #self.collect_state = jtu.tree_multimap(lambda *x: jnp.stack(x), *self.collect_state)
        #bT_outputs = jnp.stack(bT_outputs)
        bT_outputs = jtu.tree_map(lambda *x: jnp.stack(x), *bT_outputs) 
        new_self = self.replace(collect_idx=self.collect_idx + 1, collect_state=self.collect_state) 
        return new_self, bT_outputs
 