import casadi as ca
import horizon.problem as prb
import ipdb
import jax_f16.f16 as jf16
import numpy as np
from casadi_f16.f16 import F16
from horizon import variables as sv

from clfrl.dyn.dyn_types import State
from clfrl.dyn.f16_gcas import A_BOUNDS, B_BOUNDS, F16GCAS
from clfrl.dyn_cs.task_cs import TaskCS
from clfrl.mpc.mpc import set_bounds


class F16GCASCS(TaskCS):
    TaskCls = F16GCAS

    def __init__(self):
        self._task = F16GCAS()
        self._f16 = F16()
        self.thrtl_nom = F16.trim_control()[F16.THRTL]

        self._x_scale = jf16.F16.state(
            100.0, [1.0, 1.0], [0.2, 0.2, 0.2], [1.0, 1.0, 1.0], [1_000, 1_000, 500], 50.0, [1.0, 1.0, 1.0]
        )
        # self._x_scale = np.ones(self.nx)
        # self._x_scale[F16.PN] = 1e3
        # self._x_scale[F16.PE] = 1e3
        # self._x_scale[F16.H] = 100.0

    @property
    def x_scale(self):
        return self._x_scale

    @property
    def task(self) -> F16GCAS:
        return self._task

    def xdot(self, state: sv.StateVariable, control: sv.InputVariable, scale: bool = False):
        # Add throttle.
        real_state = state
        if scale:
            real_state = state * self._x_scale

        control_full = ca.vertcat(control, self.thrtl_nom)
        real_xdot = self._f16.xdot(real_state, control_full)

        xdot = real_xdot
        if scale:
            xdot = real_xdot / self._x_scale
        return xdot

    def add_constraints(self, state: sv.StateVariable, control: sv.InputVariable, prob: prb.Problem, buffer: float):
        a_buf, b_buf = 5e-2, 5e-2
        # Alpha, beta.
        set_bounds(
            state[F16.ALPHA], A_BOUNDS[0] + a_buf + buffer, A_BOUNDS[1] - a_buf - buffer, self._x_scale[F16.ALPHA]
        )
        set_bounds(state[F16.BETA], B_BOUNDS[0] + b_buf + buffer, B_BOUNDS[1] - b_buf - buffer, self._x_scale[F16.BETA])
        # Altitude.
        set_bounds(state[F16.H], 0.0 + 10 * buffer, self._task._alt_max - 10 * buffer, self._x_scale[F16.H])

    def nominal_val_state(self) -> State:
        return self._task.nominal_val_state()
