import functools as ft
import collections
from typing import Tuple, Union, List, NamedTuple, Optional

import numpy as np
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import jax.random as jrd
from jax import lax
from flax import struct

from efppo.task.task import TaskState
from efppo.task.dyn_types import TControl, THFloat, TObs, TDone
from efppo.utils.jax_types import IntScalar, TFloat
from efppo.utils.rng import PRNGKey 
from efppo.utils.jax_utils import merge01

 


Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations'])


class Experience(NamedTuple):
    Tp1_state: TaskState
    Tp1_nxt_state: TaskState
    Tp1_obs: TObs
    Tp1_nxt_obs: TObs
    Tp1_z: TFloat
    Tp1_nxt_z: TFloat
    T_control: TControl
    T_logprob: TFloat
    T_l: TFloat
    Th_h: THFloat
    T_done: TDone
    T_expert_control: TControl
 
@dataclass
class ReplayBuffer:
    _key: PRNGKey
    _capacity: int = 10240 
    _offsets: IntScalar = jnp.asarray([0])
    _dangling: bool = False
    experiences: Optional[Experience] = None
    
    
    @classmethod
    def create(cls, key: PRNGKey, capacity: Optional[int] = None):
        cls._key = key
        cls._capacity = capacity
        return cls(_key = key, _capacity = capacity)

    @property
    def size(self):
        return self._offsets.sum()
    
    def insert(self, rollout):        
        cur_experiences = self.experiences
        cur_offsets = self._offsets
        cur_dangling = self._dangling

        if cur_experiences is None:
            cur_experiences = Experience(
                Tp1_state = jnp.zeros((0, rollout.Tp1_state.shape[-1]), dtype=rollout.Tp1_state.dtype), 
                Tp1_nxt_state = jnp.zeros((0, rollout.Tp1_state.shape[-1]), dtype=rollout.Tp1_state.dtype), 
                Tp1_obs = jnp.zeros((0, rollout.Tp1_obs.shape[-1]), dtype=rollout.Tp1_obs.dtype), 
                Tp1_nxt_obs = jnp.zeros((0, rollout.Tp1_obs.shape[-1]), dtype=rollout.Tp1_obs.dtype), 
                Tp1_z = jnp.zeros((0), dtype=rollout.Tp1_z.dtype), 
                Tp1_nxt_z = jnp.zeros((0), dtype=rollout.Tp1_z.dtype), 
                T_control = jnp.zeros((0), dtype=rollout.T_control.dtype),
                T_logprob = jnp.zeros((0), dtype=rollout.T_logprob.dtype), 
                T_l = jnp.zeros((0), dtype=rollout.T_l.dtype), 
                Th_h = jnp.zeros((0, rollout.Th_h.shape[-1]), dtype=rollout.Th_h.dtype), 
                T_done = jnp.zeros((0), dtype=rollout.T_done.dtype),
                T_expert_control = jnp.zeros((0), dtype=rollout.T_control.dtype)
                )
            cur_offsets = jnp.zeros((0)) 
            cur_dangling = False
 

        new_experiences = Experience(
            Tp1_state = merge01(rollout.Tp1_state[:,:-1]), 
            Tp1_nxt_state = merge01(rollout.Tp1_state[:,1:]), 
            Tp1_obs = merge01(rollout.Tp1_obs[:,:-1]), 
            Tp1_nxt_obs = merge01(rollout.Tp1_obs[:,1:]), 
            Tp1_z = merge01(rollout.Tp1_z[:,:-1]), 
            Tp1_nxt_z = merge01(rollout.Tp1_z[:,1:]), 
            T_control = merge01(rollout.T_control),
            T_logprob = merge01(rollout.T_logprob), 
            T_l = merge01(rollout.T_l), 
            Th_h = merge01(rollout.Th_h), 
            T_done = merge01(rollout.T_done),
            T_expert_control = merge01(rollout.T_expert_control)
            )
        # Add the last experience time + 1 as if it is an initial step for the nxt
        init_ts = jnp.asarray([new_experiences.T_done.shape[0]]).astype(int)
        if jnp.any(new_experiences.T_done[:-1]) > 0: 
            # Get the time step where done == 1 and add 1 to the time step to get the initial step for the next state
            init_ts = jnp.concatenate((jnp.where(new_experiences.T_done[:-1] > 0)[0] + 1, init_ts), axis= 0).astype(int)
         
    
        def extract_experience_from_init_ts(init_t, nxt_init_ts, pre_experiences):
            traj_lens = []
            for nxt_init_t in nxt_init_ts:
                traj_len = nxt_init_t - 1 - init_t
                nxt_experiences = jtu.tree_map(
                    lambda x,y: jnp.concatenate((x, lax.dynamic_slice_in_dim(y, init_t, traj_len)), axis = 0), 
                    pre_experiences, new_experiences
                    )
                init_t = nxt_init_t
                pre_experiences = nxt_experiences
                traj_lens.append(traj_len)
            return nxt_experiences, jnp.asarray(traj_lens)
          

        self._experiences, self._offsets = extract_experience_from_init_ts(0, init_ts, cur_experiences)
        if cur_dangling:               
            self._offsets = jnp.concatenate((cur_offsets.at[-1].set(cur_offsets[-1] + self._offsets[0]), self._offsets[1:]), axis = 0)
        elif cur_offsets.shape[0] > 0:
            self._offsets = jnp.concatenate((cur_offsets, self._offsets))
         
        
        self._dangling = new_experiences.T_done[-1] > 0
    
        while self.size > self._capacity and (self._offsets.shape[0] > 1 or self._dangling):
            self.experiences = jtu.tree_map(
                    lambda x: lax.dynamic_slice_in_dim(x, self._offsets[1], x.shape[0]), 
                    self.experiences
                )
            self._offsets = self._offsets[1:]
   
    def sample(self, num_batches: int, batch_size: int) -> Experience:
        experiences = self.experiences
        replace = batch_size >= self.size
        def sample_one_batch(_):
            return jtu.tree_map(lambda x: jrd.choice(self._key, x, (batch_size,), axis = 0, replace = replace), experiences)
        b_experiences = jax.vmap(sample_one_batch)(jnp.arange(num_batches))
        return b_experiences