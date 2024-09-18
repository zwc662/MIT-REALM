import functools as ft
from typing import Any, NamedTuple, Tuple, Optional, Callable, Dict

import numpy as np

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from attrs import define
from flax import struct
 
from efppo.task.task import Task, TaskState
from efppo.task.dyn_types import TControl, THFloat, TObs, TDone, Obs
from efppo.utils.jax_types import FloatScalar, IntScalar, TFloat
from efppo.utils.cfg_utils import Cfg 
from efppo.utils.jax_utils import concat_at_front, concat_at_end
from efppo.utils.rng import PRNGKey
from efppo.utils.tfp import tfd 


 
class RolloutOutput(NamedTuple):
    Tp1_state: TaskState
    Tp1_obs: TObs
    Tp1_z: TFloat
    T_control: TControl
    T_logprob: TFloat
    T_l: TFloat
    Th_h: THFloat
    T_done: TDone
    T_expert_control: TControl
    T_agent_control: TControl


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
        agent_control = get_pol(obs_pol, state.z)
        expert_control = task.get_expert(state.state, control)
        envstate_new, control = task.step(state.state, agent_control)

        # Z dynamics.
        l = task.l(envstate_new, control)
        
        z_new = (state.z - l) / disc_gamma
        z_new = jnp.clip(z_new, z_min, z_max)
        if hasattr(task, 'l1_control_info'):
            task.l1_control_info['z'] = z_new

        return CollectorState(state.steps, envstate_new, z_new), (envstate_new, obs_pol, z_new, control, expert_control, agent_control)
    if hasattr(task, 'l1_control_info'):
        task.l1_control_info['z'] = z0
    colstate0 = CollectorState(0, x0, z0)
    collect_state, (T_envstate, T_obs, T_z, T_u, T_expert_u, T_agent_u) = lax.scan(_body, colstate0, None, length=rollout_T)
    obs_final = task.get_obs(collect_state.state)

    # Add the initial observations.
    Tp1_state = jtu.tree_map(concat_at_front, colstate0.state, T_envstate)
    T_state_fr, T_state_to = Tp1_state[:-1], Tp1_state[1:]
    Tp1_obs = jtu.tree_map(concat_at_end, T_obs, obs_final)
    Tp1_z = concat_at_front(z0, T_z)
    T_done = Tp1_z * 0.
    T_l = jax.vmap(task.l)(T_state_to, T_u)
    Th_h = jax.vmap(task.h_components)(T_state_to) 

    return RolloutOutput(Tp1_state, Tp1_obs, Tp1_z, T_u, None, T_l, Th_h, T_done, T_expert_u, T_agent_u)

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
        agent_control, logprob = a_pol.experimental_sample_and_log_prob(seed=key)
        expert_control = task.get_expert(state.state, agent_control)
        envstate_new, control = task.step(state.state, agent_control)
        
        # Z dynamics.
        l = task.l(envstate_new, control) 
        z_new = (state.z - l) / disc_gamma
        z_new = jnp.clip(z_new, z_min, z_max)

        return CollectorState(state.steps, envstate_new, z_new), (envstate_new, obs_pol, z_new, control, logprob, expert_control, agent_control)

    # Randomly sample z0.
    key_z0, key_step = jr.split(key0)
    z0 = jr.uniform(key_z0, minval=z_min, maxval=z_max)
    colstate0 = colstate0._replace(z=z0)

    assert colstate0.steps.shape == tuple()
    T_keys = jr.split(key_step, rollout_T)
    collect_state, (T_envstate, T_obs, T_z, T_u, T_logprob, T_expert_u, T_agent_u) = lax.scan(_body, colstate0, T_keys, length=rollout_T)
    collect_state = collect_state._replace(steps=collect_state.steps + rollout_T)
    obs_final = task.get_obs(collect_state.state)

    # Add the initial observations.
    Tp1_state = jtu.tree_map(concat_at_front, colstate0.state, T_envstate)
    T_state_fr, T_state_to = Tp1_state[:-1], Tp1_state[1:]
    Tp1_obs = jtu.tree_map(concat_at_end, T_obs, obs_final)
    Tp1_z = concat_at_front(z0, T_z)
    T_l = jax.vmap(task.l)(T_state_to, T_u)
    Th_h = jax.vmap(task.h_components)(T_state_to)
    T_done = Tp1_z * 0.

    return collect_state, RolloutOutput(Tp1_state, Tp1_obs, Tp1_z, T_u, T_logprob, T_l, Th_h, T_done, T_expert_u, T_agent_u)

def collect_single_env_mode(
    task: Task,
    x0: TaskState,
    z0: float,
    get_pol: Callable[[Obs, FloatScalar], Dict[str, Any]],
    disc_gamma: float,
    z_min: float,
    z_max: float,
    rollout_T: int,
    verbose: bool = False,
    soft_reset: bool = False
):
    def _body(state: CollectorState, _):
        obs_pol = task.get_obs(state.state)
        #a_pol: tfd.Distribution = get_pol(obs_pol, state.z)
        #control = a_pol.mode()
        agent_control = get_pol(obs_pol, state.z)
        expert_control = task.get_expert(state.state, **agent_control)
        envstate_new, control = task.step(state.state, **agent_control)
 
        # Z dynamics.
        l = task.l(envstate_new, control)
        h = task.h_components(envstate_new)
        z_new = (state.z - l) / disc_gamma
        z_new = jnp.clip(z_new, z_min, z_max)
        new_state = CollectorState(state.steps + 1, envstate_new.reshape(-1), z_new)
        return new_state, (envstate_new, obs_pol, z_new, l, h, control, expert_control, agent_control)
 
    collect_state = CollectorState(0, x0, z0)
    
    T_envstate = []
    T_obs = []
    T_z = []
    T_u = []
    T_l = []
    Th_h = [] 
    T_done = []
    T_expert_u = []
    T_agent_u = []

    for t in range(rollout_T):
        collect_state, (envstate_new, obs_pol, z_new, l, h, control, expert_control, agent_control) = _body(collect_state, None)

        assert not np.any(np.isnan(envstate_new))
        assert not np.any(np.isnan(obs_pol))
    
        if jnp.any(jnp.logical_or(jnp.isinf(control), jnp.isnan(control))):
            print("[NaN control Warning]")
        if verbose:
            print(f"Step: {task.cur_step} | State: {task.cur_state} | Control: {control} | Actuator: {task.cur_action} | l: {l} | h: {h}")
            #control = task.cur_action.reshape(control.shape)
        
        
        T_obs.append(obs_pol)
        T_z.append(z_new)
        T_u.append(control) 
        T_l.append(l)
        Th_h.append(h)
        T_expert_u.append(expert_control)
        T_agent_u.append(agent_control)
        

        if task.should_reset(envstate_new):
            # Add done to the data collection used as mask
            collect_state = collect_state._replace(
                steps = collect_state.steps,
                state = task.reset(mode='soft' if soft_reset else None),
                z=collect_state.z
                )
            T_envstate.append(collect_state.state)
            T_done.append(1.0)
        else:
            T_envstate.append(envstate_new)
            T_done.append(0.0)

    T_done = [0] + T_done
    T_envstate = [x0] + T_envstate
    T_z = [z0] + T_z 
    obs_final = task.get_obs(collect_state.state)
    T_obs += [obs_final]
    
 
    collect_state = collect_state._replace(steps = rollout_T - 1)
     

    Tp1_state = jnp.stack(T_envstate)
    T_u = jnp.stack(T_u)
    Tp1_obs = jnp.stack(T_obs)
    Tp1_z = jnp.stack(T_z)
    T_l = jnp.stack(T_l)
    Th_h = jnp.stack(Th_h)
    T_done = jnp.stack(T_done)
    T_expert_u = jnp.stack(T_expert_u)
    T_agent_u = jnp.stack(T_agent_u)
    '''
    # Add the initial observations.
    Tp1_state = jnp.stack((x0, *T_envstate))
    T_state_fr, T_state_to = Tp1_state[:-1], Tp1_state[1:]
    T_u = jnp.stack((*T_u, jnp.zeros(task.nu)))
    Tp1_obs = jnp.stack((*T_obs, obs_final)) 
    Tp1_z = jnp.concatenate((jnp.asarray([z0]), jnp.asarray(T_z))).reshape(-1)
    T_l = jnp.concatenate((-jnp.ones(1), jnp.asarray(T_l))).reshape(-1)
    Th_h = jnp.stack((-jnp.ones(len(task.h_labels)), *Th_h)).reshape(-1, len(task.h_labels))
    '''
                     
    return RolloutOutput(Tp1_state, Tp1_obs, Tp1_z, T_u, None, T_l, Th_h, T_done, T_expert_u, T_agent_u)

def collect_single_batch(
    task: Task,
    key0: PRNGKey,
    colstate0: CollectorState,
    get_pol,
    disc_gamma: float,
    z_min: float,
    z_max: float,
    rollout_T: int,
    max_T: float = float('inf')
):
    def _body(state: CollectorState, key):
        obs_pol = task.get_obs(state.state)
        #a_pol: tfd.Distribution = get_pol(obs_pol, state.z.squeeze())
        agent_control, logprob = get_pol(obs_pol, state.z.squeeze(), key = key)
        expert_control = task.get_expert(state.state, agent_control)
        '''
        s = tfd.Sample(
            tfd.MultivariateNormalDiag(loc=[0., 0.], scale_diag=[1, 1.]), sample_shape=1
            )
        base_control = s.sample([1], seed = key)
        logprob = s.log_prob(base_control).reshape(logprob.shape)
        control = (a_pol.scale @ base_control.reshape(-1, 1) + a_pol.loc.reshape(-1, 1)).reshape(control.shape)
        '''
        envstate_new, control = task.step(state.state, np.asarray(agent_control).reshape(-1))
    
        # Z dynamics.
        l = task.l(envstate_new, control)
        
        h = task.h_components(envstate_new)
        z_new = (state.z - l) / disc_gamma
        z_new = jnp.clip(z_new, z_min, z_max)
 
        new_state = CollectorState(state.steps + 1, envstate_new.reshape(-1), z_new)
        return new_state, (envstate_new, obs_pol, z_new, l, h, control, logprob, expert_control, agent_control)
    
    fst_step, x0, z0 = colstate0.steps, colstate0.state, colstate0.z
    collect_state = CollectorState(fst_step, x0, z0)
         
    T_keys = jr.split(key0, rollout_T)

    # Initialize outputs for the loop
    T_envstate = []
    T_obs = []
    T_z = []
    T_u = []
    T_logprob = []
    T_l = []
    Th_h = []
    T_done = []
    T_expert_u = []
    T_agent_u = []

    collect_state = colstate0 
    # Perform the loop over rollout_T
    for t in range(rollout_T):     
        assert not jnp.isnan(collect_state.state).any(), f'{collect_state.state=}'   
        collect_state, (envstate_new, obs_pol, z_new, l, h, control, logprob, expert_control, agent_control) = _body(collect_state, T_keys[t])
        assert not jnp.isnan(envstate_new).any(), f'{task.cur_state=}'
        assert not jnp.isnan(obs_pol).any(), f'{task.cur_state=}'
        #assert not jnp.isnan(control).any(), f'{control=}'
        assert not jnp.any(jnp.logical_or(jnp.isinf(control), jnp.isnan(control))) or jnp.any(jnp.logical_or(jnp.isnan(logprob), jnp.isinf(logprob))), f"[NaN control Warning] Step: {task.cur_step} | State: {task.cur_state} | Control: {control} | Actuator: {task.cur_action} | logprob: {logprob} | l: {l} | h: {h}"
            #control = task.cur_action.reshape(control.shape[0], task.nu)
        
        #assert not jnp.isnan(logprob).any(), f'{logprob=}'
        
        
        T_obs.append(obs_pol)
        T_z.append(z_new)
        T_u.append(control)
        T_logprob.append(logprob)
        T_l.append(l)
        Th_h.append(h)
        T_expert_u.append(expert_control)
        T_agent_u.append(agent_control)

        shouldreset = task.should_reset(envstate_new) or (collect_state.steps >= max_T)
        if shouldreset:
            #print("Did reset")
            collect_state = collect_state._replace(
                steps = 0,
                state = task.reset(mode = 'train'),
                z=colstate0.z
                )
            T_done.append(1.0)
            T_envstate.append(collect_state.state)
        else:
            T_done.append(0.0)
            T_envstate.append(envstate_new)
            
    #print(f"Single batch sampled control {len(T_u)=}, {len(T_logprob)=}, {np.mean(T_logprob)=}")
    # Convert lists to jnp arrays
    T_envstate = jnp.stack(T_envstate)
    T_obs = jnp.stack(T_obs)
    T_z = jnp.stack(T_z)
    T_u = jnp.stack(T_u)
    T_logprob = jnp.stack(T_logprob)
    T_l = jnp.stack(T_l)
    Th_h = jnp.stack(Th_h)
    T_done = jnp.stack(T_done)
    T_expert_u = jnp.stack(T_expert_u)
    T_agent_u = jnp.stack(T_agent_u)
    
    obs_final = task.get_obs(collect_state.state)

    # Add the initial observations.
    Tp1_state = jnp.stack((colstate0.state, *T_envstate))
    T_state_fr, T_state_to = Tp1_state[:-1], Tp1_state[1:]
    Tp1_obs = jnp.stack((*T_obs, obs_final)) 
    Tp1_z = jnp.concatenate((jnp.asarray([z0]), jnp.asarray(T_z))).reshape(-1)
    #T_l = jnp.concatenate((jnp.asarray([0]), jnp.asarray(T_l))).reshape(-1)
    #Th_h = jnp.stack((jnp.asarray([-1]), *Th_h)).reshape(-1, len(task.h_labels))
    
    return collect_state, RolloutOutput(Tp1_state, Tp1_obs, Tp1_z, T_u, T_logprob,T_l, Th_h, T_done, T_expert_u, T_agent_u)

class Collector(struct.PyTreeNode):
    collect_idx: int
    key: PRNGKey
    collect_state: CollectorState
    task: Task = struct.field(pytree_node=False)
    cfg: CollectorCfg = struct.field(pytree_node=False)
    
 
    @classmethod
    def create(cls, key: PRNGKey, task: Task, cfg: CollectorCfg):
        key, key_init = jr.split(key)
        b_state = jnp.asarray(task.sample_x0_train(key_init, cfg.n_envs))
        b_steps = jnp.zeros(cfg.n_envs, dtype=jnp.int32)
        b_z0 = jnp.zeros(cfg.n_envs)
        collector_state = CollectorState(b_steps, b_state, b_z0) 
        return Collector(0, key, collector_state, task, cfg)
    
    def reset(self):
        b_state = jnp.asarray(self.task.sample_x0_train(self.key, self.cfg.n_envs))
        b_steps = jnp.zeros(self.cfg.n_envs, dtype=jnp.int32)
        b_z0 = jnp.zeros(self.cfg.n_envs)
        collect_state = CollectorState(b_steps, b_state, b_z0) 
        return self.replace(collect_idx=self.collect_idx + 1, collect_state=collect_state)
    
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

        def reset_fn(should_reset, state_reset_new, state_reset_old, steps_old, dones_old):
            def reset_fn_(arr_new, arr_old):
                return jnp.where(should_reset, arr_new, arr_old)

            state_new = jtu.tree_map(reset_fn_, state_reset_new, state_reset_old)
            steps_new = jnp.where(should_reset, 0, steps_old)
            dones_new = jnp.where(should_reset, 1., dones_old)
            
            return steps_new, state_new

        b_steps_new, b_state_new, b_done = jax.vmap(reset_fn)(
            b_shouldreset, b_state_reset, collect_state.state, collect_state.steps, bT_outputs.dones
        )
        bT_outputs.T_done = bT_outputs.T_done.at[-1].set(b_done)
        collect_state = CollectorState(b_steps_new, b_state_new, collect_state.z)

        new_self = self.replace(collect_idx=self.collect_idx + 1, collect_state=collect_state)
        return new_self, bT_outputs

    def _collect_single_batch(
        self, env_idx: int, key0: PRNGKey, colstate0: CollectorState, get_pol, disc_gamma: float, z_min: float, z_max: float, rollout_T: Optional[int] = None
    ) -> tuple[CollectorState, RolloutOutput]: 
        if rollout_T is None:
            if hasattr(self.cfg, 'rollout_T'):
                rollout_T = self.cfg.rollout_T
            else:
                rollout_T = 1
        
        max_T = float('inf')
        if hasattr(self.cfg, 'max_T'):
            max_T = self.cfg.max_T

        return collect_single_batch(self.task, key0, colstate0, get_pol, disc_gamma, z_min, z_max, rollout_T, max_T)


    def collect_batch_iteratively(
        self, get_pol, disc_gamma: float, z_min: float, z_max: float, rollout_T: int = 1
    ) -> tuple["Collector", RolloutOutput]:
        key0 = jr.fold_in(self.key, self.collect_idx)
        key_pol, key_reset_bernoulli, key_reset = jr.split(key0, 3)
        b_keys = jr.split(key_pol, self.cfg.n_envs)
        
        # Initialize lists to accumulate results
        bT_outputs = []
        # Use a for loop instead of jax.vmap
 
        p_reset = self.p_reset
        key_reset = key_reset

        
        for i in range(self.cfg.n_envs):
            collect_state_i = jtu.tree_map(lambda x: x[i], self.collect_state)
            
            '''
            print(f"Env {i} / {self.cfg.n_envs}")
            shouldreset = jr.bernoulli(key_reset_bernoulli, p_reset)
            shouldreset = jnp.logical_or(shouldreset, collect_state_i.steps >= max_T).item()
            shouldreset = (self.task.cur_done > 0.).any() | shouldreset | (
                i > 0 and self.task.should_reset(self.collect_state.state[i - 1])
                )
            shouldreset = (collect_state_i.steps < self.cfg.max_T)
            if shouldreset:
                random_map = False #jr.bernoulli(key_reset, p_reset)
                # Randomly sample z0.
                key_z0, key0 = jr.split(key0)
                z0 = jr.uniform(key_z0, minval=z_min, maxval=z_max)
        
                collect_state_i = collect_state_i._replace(
                    steps = 0,
                    state = self.task.reset(mode='train', random_map = random_map),
                    z=z0
                    )
            else:
                collect_state_i = collect_state_i._replace(
                    steps = self.collect_state.steps[i - 1],
                    state = self.collect_state.state[i - 1],
                    z = self.collect_state.z[i-1]
                )
            '''
            collect_state_i, bT_output = self._collect_single_batch(
                i, 
                key0, 
                collect_state_i,
                get_pol,
                disc_gamma=disc_gamma, 
                z_min=z_min, 
                z_max=z_max,
                rollout_T = rollout_T 
                )
            #print(f"Sampled control {np.mean(bT_output.T_logprob)=}")
            # Resample x0
            
            bT_outputs.append(bT_output)
            self.collect_state.state.at[i].set(collect_state_i.state)
            self.collect_state.steps.at[i].set(collect_state_i.steps)
            self.collect_state.z.at[i].set(collect_state_i.z.item())
            


        assert self.collect_state.steps.shape == (self.cfg.n_envs,)
        # Stack the collected states and outputs
        #self.collect_state = jtu.tree_multimap(lambda *x: jnp.stack(x), *self.collect_state)
        #bT_outputs = jnp.stack(bT_outputs)
        bT_outputs = jtu.tree_map(lambda *x: jnp.stack(x), *bT_outputs) 
        new_self = self.replace(collect_idx=self.collect_idx + 1, collect_state=self.collect_state) 
        return new_self, bT_outputs
 
 