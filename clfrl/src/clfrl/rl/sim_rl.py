from typing import Callable

import jax.lax as lax
import jax.random as jr

from clfrl.dyn.dyn_types import PolObs, State
from clfrl.dyn.task import Task
from clfrl.utils.jax_utils import concat_at_end, merge01
from clfrl.utils.rng import PRNGKey
from clfrl.utils.tfp import tfd

_PolicyFn = Callable[[PolObs], tfd.Distribution]


class SimMode:
    def __init__(self, task: Task, policy, T: int):
        self.task = task
        self.policy: _PolicyFn = policy
        self.T = T

    def rollout(self, x0: State):
        def body(state, _):
            Vobs, polobs = self.task.get_obs(state)
            dist = self.policy(polobs)
            control = dist.mode()
            state_new = self.task.step(state, control)
            return state_new, (state, control)

        state_f, (T_state, T_control) = lax.scan(body, x0, None, self.T)
        Tp1_state = concat_at_end(T_state, state_f, axis=0)

        return Tp1_state, T_control

    def rollout_plot(self, x0: State):
        def body(state, _):
            Vobs, polobs = self.task.get_obs(state)
            dist = self.policy(polobs)
            control = dist.mode()
            T_states = self.task.step_plot(state, control)
            state_new = T_states[-1]
            return state_new, (T_states[:-1], control)

        state_f, (TT_state, T_control) = lax.scan(body, x0, None, self.T)
        T_state = merge01(TT_state)
        Tp1_state = concat_at_end(T_state, state_f, axis=0)

        return Tp1_state, T_control


class SimRng:
    def __init__(self, task: Task, policy, T: int):
        self.task = task
        self.policy: _PolicyFn = policy
        self.T = T

    def rollout(self, key: PRNGKey, x0: State):
        def body(state, key_):
            Vobs, polobs = self.task.get_obs(state)
            dist = self.policy(polobs)
            control = dist.sample(seed=key_)
            state_new = self.task.step(state, control)
            return state_new, (state, control)

        T_keys = jr.split(key, self.T)
        state_f, (T_state, T_control) = lax.scan(body, x0, T_keys, self.T)
        Tp1_state = concat_at_end(T_state, state_f, axis=0)

        return Tp1_state, T_control


class SimModeRob:
    def __init__(self, task: Task, policy_u, policy_d, T: int):
        self.task = task
        self.policy_u: _PolicyFn = policy_u
        self.policy_d: _PolicyFn = policy_d
        self.T = T

    def rollout(self, x0: State):
        def body(state, _):
            Vobs, polobs = self.task.get_obs(state)
            dist_u, dist_d = self.policy_u(polobs), self.policy_d(Vobs)
            control, disturb = dist_u.mode(), dist_d.mode()
            state_new = self.task.step(state, control, disturb)
            return state_new, (state, control, disturb)

        state_f, (T_state, T_control, T_disturb) = lax.scan(body, x0, None, self.T)
        Tp1_state = concat_at_end(T_state, state_f, axis=0)

        return Tp1_state, T_control, T_disturb


class SimRngRob:
    def __init__(self, task: Task, policy_u, policy_d, T: int):
        self.task = task
        self.policy_u: _PolicyFn = policy_u
        self.policy_d: _PolicyFn = policy_d
        self.T = T

    def rollout(self, key: PRNGKey, x0: State):
        def body(state, key_):
            key_u, key_d = jr.split(key_, 2)
            Vobs, polobs = self.task.get_obs(state)
            dist_u, dist_d = self.policy_u(polobs), self.policy_d(Vobs)
            control, disturb = dist_u.sample(seed=key_u), dist_d.sample(seed=key_d)
            state_new = self.task.step(state, control, disturb)
            return state_new, (state, control, disturb)

        T_keys = jr.split(key, self.T)
        state_f, (T_state, T_control, T_disturb) = lax.scan(body, x0, T_keys, self.T)
        Tp1_state = concat_at_end(T_state, state_f, axis=0)

        return Tp1_state, T_control, T_disturb
