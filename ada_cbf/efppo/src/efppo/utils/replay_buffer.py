import functools as ft
import collections
from typing import Tuple, Union, List, NamedTuple, Optional

import numpy as np

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import lax

from flax import struct
 
from efppo.task.task import TaskState
from efppo.task.dyn_types import TControl, THFloat, TObs, TDone
from efppo.utils.jax_types import IntScalar, TFloat
from efppo.utils.rng import PRNGKey
from efppo.rl.collector import RolloutOutput

from dataclasses import dataclass


Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations'])


class Experience(NamedTuple):
    Tp1_state: TaskState
    Tp1_nxt_state: TaskState
    Tp1_obs: TObs
    Tp1_nxt_obs: TObs
    Tp1_z: TFloat
    T_control: TControl
    T_logprob: TFloat
    T_l: TFloat
    Th_h: THFloat
    T_done: TDone


class Experience_Node(dataclass):
    cur_experience: Experience
    hindsight_experiences: Optional[Experience] = None

    @classmethod
    def create(cls, experience: Experience):
        fst_done_idx = jnp.select(experience.T_done == 1, jnp.arange(experience.T_done.shape[0]))[0]
        nxt_init_idx = fst_done_idx + 1
        cur_experience = jtu.tree_map(lambda x: x[:nxt_init_idx], experience)
        if nxt_init_idx > experience.T_done.shape[0] - 1:
            return cls(cur_experience)
        else:
            hindsight_experience = jtu.tree_map(lambda x: x[nxt_init_idx:], experience)
            return cls(cur_experience, hindsight_experience)

def get_current(experiences: Experience_Node):
    return experiences.cur_experience
 
def update_current(experiences: Experience_Node, new_experience) -> Experience_Node:
    experiences.cur_experience = new_experience
    return experiences

def add_new(new_experience: Experience, hindsight_experiences: Optional[Experience] = None) -> Experience_Node:
    return Experience_Node(new_experience = new_experience, hindsight_experience = hindsight_experiences)

def pop_oldest(experiences: Experience_Node) -> Experience_Node:
    if experiences.hindsight_experiences is None:
        return None, None
    if experiences.hindsight_experiences.hindsight_experiences is None:
        return None, Experience_Node(cur_experience = experiences.cur_experience)
    oldest_experience, remaining_hindsight_experiences = pop_oldest(experiences.hindsight_experiences)
    return oldest_experience, Experience_Node(cur_experience = experiences.cur_experience, hindsight_experience = remaining_hindsight_experiences)
 

class ReplayBuffer_v0(dataclass):
    _key: PRNGKey
    _capacity: int = 10240
    size: int = 0
     
    init_ids: IntScalar = [0] 
    hindsight_experiences: Optional[Experience_Node]
    cur_experience: Optional[Experience]
    experiences: Optional[Experience]
    
    @classmethod
    def create(cls, key: PRNGKey, capacity: Optional[int] = None, **kwargs):
        cls._key = key
        cls._capacity = capacity
        return cls(_key = key, _capacity = capacity, **kwargs)
 
    def insert(self, rollout): 

        Tp1_state = rollout.Tp1_state[:-1],
        Tp1_nxt_state = jnp.where(rollout.T_done, rollout.Tp1_state[:-1], rollout.Tp1_state[1:])
        Tp1_obs = rollout.Tp1_obs[:-1],
        Tp1_nxt_obs = jnp.where(rollout.T_done, rollout.Tp1_obs[:-1], rollout.Tp1_obs[1:]) 
        Tp1_z = rollout.Tp1_z,
        T_control = rollout.T_control,
        T_logprob = rollout.T_logprob,
        T_l = rollout.T_l,
        Th_h = rollout.Th_h,
        T_done = rollout.T_done
        
        experiences = Experience(
                    Tp1_state,
                    Tp1_nxt_state,
                    Tp1_obs,
                    Tp1_nxt_obs,
                    Tp1_z,
                    T_control,
                    T_logprob,
                    T_l,
                    Th_h,
                    T_done
                )
        
        init_ids = 1 + jnp.select(T_done > 0, jnp.arange(T_done.shape[0]), experiences)



        ## Process the headless first experience among the 'experiences' by concatenating it with 'self.cur_experience' and then 'self.hindsight_experiences'
        cur_experience = None
        if init_ids[0] > 0:
            cur_experience = jtu.tree_map(lambda x: x[:init_ids[0]], experiences)
        else:
            assert self.cur_experience is None

        if self.cur_experience is not None:
            assert cur_experience is not None 
            self.cur_experience = jtu.tree_map(lambda x, y: jnp.concatenate((x, y), axis = 0), self.cur_experience, cur_experience)
            self.hindsight_experiences = add_new(self.cur_experience, self.hindsight_experiences)
            cur_experience = None
            self.cur_experience = None
        
        ## Process the tailless experience in 'experiences' by assigning it as the new self.cur_experience
        if init_ids[-1] < T_done.shape[0]:
            self.cur_experience = jtu.tree_map(lambda x: x[init_ids:], experiences)

        ## Process the rest
        experience_ids_lst = [jnp.arange(from_idx, to_idx) for from_idx, to_idx in zip(init_ids[:-1], init_ids[1:])]
        for experience_ids in experience_ids_lst:
            self.hindsight_experiences = add_new(jtu.tree_map(lambda x: x[experience_ids], experiences), self.hindsight_experiences)
        

        ## Remove oldest experiences until within self.capacity
        self.size += T_done.shape[0]
        if self.size >= self._capacity:
            print('Reach capacity. Removing Experiences')
        
        while self.size.shape[0] > 0:
            oldest_experience, self.hindsight_experiences = pop_oldest(self.hindsight_experiences)
            self.size -= oldest_experience.T_done.shape[0]
            
        def serialize(experience, experiences):
            if experiences is None:
                return experience
            
            two_experience = jtu.tree_map(lambda x, y: jnp.concatenate((x, y), axis = 0), experience, experiences.cur_experience)
            if experiences.hindsight_experiences is None:
                return two_experience
            return serialize(two_experience, experiences.hindsight_experiences)

        if self.cur_experince is not None:
            self.experiences = jtu.tree_map(serialize, self.cur_experience, self.hindsight_experiences)
        else:
            self.experiences = jtu.tree_map(serialize, self.hindsight_experiences.experience, self.hindsisght_experiences.hindsight_experiences)



    @ft.partial(jax.jit, static_argnums=(1,))
    def sample(self, size: int) -> RolloutOutput:
        sample_key, self.key = jax.random.split(self.key)
        selected_ids = jnp.random.choice(sample_key, jnp.arange(self.size), size=size)
        return jtu.tree_map(lambda x: x[selected_ids], self.experiences)
    
 
class ReplayBuffer(struct.PyTreeNode):
    _key: PRNGKey
    _capacity: int = 10240
    _offsets: IntScalar = jnp.asarray([0])
    _dangeling: bool
    experiences: Optional[Experience]
    
    
    @classmethod
    def create(cls, key: PRNGKey, capacity: Optional[int] = None):
        cls._key = key
        cls._capacity = capacity
        return cls(_key = key, _capacity = capacity)

    @property
    def size(self):
        size, _ = lax.scan(
            f = lambda carry, rollout: (carry + rollout.T_done.shape[0], ), 
            init = 0,
            xs = self.rollouts
        )
        return size
    
    def insert(self, rollout): 
        # Find the indices where b is 1
        
        Tp1_state = jnp.zeros((0, *rollout.Tp1_state.shape[1:]), dtype=rollout.Tp1_state.dtype)
        Tp1_nxt_state = jnp.zeros((0, *rollout.Tp1_state.shape[1:]), dtype=rollout.Tp1_state.dtype)
        Tp1_obs = jnp.zeros((0, *rollout.Tp1_obs.shape[1:]), dtype=rollout.Tp1_obs.dtype)
        Tp1_nxt_obs = jnp.zeros((0, *rollout.Tp1_obs.shape[1:]), dtype=rollout.Tp1_obs.dtype)
        Tp1_z = jnp.zeros((0, *rollout.Tp1_z.shape[1:]), dtype=rollout.Tp1_z.dtype)
        T_control = jnp.zeros((0, *rollout.T_control.shape[1:]), dtype=rollout.T_control.dtype)
        T_logprob = jnp.zeros((0, *rollout.T_logprob.shape[1:]), dtype=rollout.T_logprob.dtype)
        T_l = jnp.zeros((0, *rollout.T_l.shape[1:]), dtype=rollout.T_l.dtype)
        Th_h = jnp.zeros((0, *rollout.Th_h.shape[1:]), dtype=rollout.Th_h.dtype)
        T_done = jnp.zeros((0, *rollout.T_done.shape[1:]), dtype=rollout.T_done.dtype)
        
        cur_experiences = lax.cond(self.experiences is None, Experience(Tp1_state = Tp1_state, Tp1_nxt_state = Tp1_nxt_state, Tp1_obs = Tp1_obs, Tp1_nxt_obs = Tp1_nxt_obs, Tp1_z = Tp1_z, T_control = T_control,
                                        T_logprob = T_logprob, T_l = T_l, Th_h = Th_h, T_done = T_done), self.experiences)
        cur_offsets = lax.cond(self.experience is None, jnp.zeros((0)), self._offsets)
        cur_dangling = lax.cond(self.experience is None, False, self._dangling)
 

        new_experiences = Experience(
            Tp1_state = rollout.Tp1_state[:-1], 
            Tp1_nxt_state = rollout.Tp1_state[1:], 
            Tp1_obs = rollout.Tp1_obs[:-1], 
            Tp1_nxt_obs = rollout.Tp1_obs[1:], 
            Tp1_z = rollout.Tp1_z, 
            T_control = rollout.T_control,
            T_logprob = rollout.T_logprob, 
            T_l = rollout.T_l, 
            Th_h = rollout.Th_h, 
            T_done = rollout.T_done
            )

        last_ts = jnp.select(rollout.T_done == 1)[0]
        init_ts = last_ts[jnp.select(last_ts < rollout.T_done.shape[0] - 1)] + 1 
        init_ts = lax.dynamic_index_in_dim(init_ts, jnp.asarray([rollout.T_done.shape[0]]), init_ts.shape[0], 0)
         
    
        def extract_experience_from_init_ts(init_ts):
            def extract_experience(carry, nxt_init_t_idx):
                init_t, pre_experiences = carry
                nxt_init_t = init_ts[nxt_init_t_idx]
                traj_len = nxt_init_t - 1 - init_t
                nxt_experiences = jtu.tree_map(lambda x,y: jnp.concatenate((x, y[init_t: nxt_init_t - 1]), axis = 0), pre_experiences, new_experiences)
                return (nxt_init_t, nxt_experiences), traj_len
            return jax.jit(extract_experience)
        
        (_, new_experiences), new_offsets_ = extract_experience_from_init_ts(init_ts)((0, cur_experiences), jnp.arange(init_ts.shape[0]))
                          
        new_offsets = lax.cond(
            cur_dangling, 
            jax.concatenate((cur_offsets.at[-1].set(cur_offsets[-1] + new_offsets_[0]), new_offsets_[1:]), axis = 0),
            jax.concatenate((cur_offsets, new_offsets_))
            )
        
        new_dangling = rollout.T_done[-1] > 0
 
        new_size = jnp.sum(new_offsets)

        new_self = self.replace(_key = self._key, _capcity = self._capacity, _size = new_size, _dangling = new_dangling, experiences = new_experiences)
 
        new_self = lax.while_loop(
            cond_fun = new_self._size > new_self._capacity,
            body_fun = lambda obj: obj.replace(
                _key = obj._key, 
                _capcity = obj._capacity, 
                _size = obj.size - obj.offsets[0], 
                _dangling = obj._dangling,
                _offset = obj.offset[1:],
                experiences = jtu.tree_map(lambda x: x[obj.offsets[1]:], obj.experience) 
            ),
            init_val = new_self
        )        

        return new_self
    

    
    