from typing import Callable

import jax.lax as lax

from clfrl.dyn.dyn_types import Control, State
from clfrl.dyn.task import Task
from clfrl.utils.jax_utils import concat_at_end, merge01

_PolicyFn = Callable[[State], Control]


class SimNCLF:
    def __init__(self, task: Task, policy, T: int):
        self.task = task
        self.policy: _PolicyFn = policy
        self.T = T

    def rollout(self, x0: State):
        def body(state, _):
            control = self.policy(state)
            state_new = self.task.step(state, control)
            return state_new, (state, control)

        state_f, (T_state, T_control) = lax.scan(body, x0, None, self.T)
        Tp1_state = concat_at_end(T_state, state_f, axis=0)

        return Tp1_state, T_control

    def rollout_plot(self, x0: State):
        def body(t_state, _):
            t, state = t_state
            control = self.policy(state)
            T_states, T_t = self.task.step_plot(state, control)
            state_new = T_states[-1]
            t_new = T_t[-1]
            return (t + t_new, state_new), (T_states[:-1], control, t + T_t[:-1])

        (t_f, state_f), (TT_state, T_control, TT_ts) = lax.scan(body, (0.0, x0), None, self.T)
        T_state, T_ts = merge01(TT_state), merge01(TT_ts)
        Tp1_state = concat_at_end(T_state, state_f, axis=0)
        Tp1_ts = concat_at_end(T_ts, t_f, axis=0)

        return Tp1_state, T_control, Tp1_ts
