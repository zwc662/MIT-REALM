import functools as ft
from abc import ABC

import casadi as ca
import horizon.problem as prb
import ipdb
import numpy as np
from horizon import variables as sv

from clfrl.dyn.task import Task


class TaskCS(ABC):
    TaskCls: Task = None

    @property
    def nx(self) -> int:
        return self.TaskCls.NX

    @property
    def nu(self) -> int:
        return self.TaskCls.NU

    @property
    def task(self) -> Task:
        ...

    def get_variables(self, prob: prb.Problem, mx: bool = False):
        abstract_casadi_type = ca.MX if mx else ca.SX
        state = prob.createStateVariable("state", dim=self.nx, abstract_casadi_type=abstract_casadi_type)
        control = prob.createInputVariable("control", dim=self.nu, abstract_casadi_type=abstract_casadi_type)

        control.setBounds(lb=self.task.u_min, ub=self.task.u_max)

        return state, control

    def xdot(self, state: sv.StateVariable, control: sv.InputVariable, scale: bool = False):
        ...

    def add_constraints(self, state: sv.StateVariable, control: sv.InputVariable, prob: prb.Problem, buffer: float):
        ...

    def step_rk4(self, x1, u, dt: float):
        xdot = ft.partial(self.xdot, control=u, scale=False)

        k1 = xdot(x1)

        x2 = x1 + k1 * dt * 0.5
        k2 = xdot(x2)

        x3 = x1 + k2 * dt * 0.5
        k3 = xdot(x3)

        x4 = x1 + k3 * dt
        k4 = xdot(x4)
        return x1 + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def integrate_rk4(self, x0, T_u, dt: float):
        x_traj = [x0]
        x = x0
        for u in T_u:
            x = self.step_rk4(x, u, dt)
            x_traj.append(np.array(x).squeeze())

        Tp1_x = np.stack(x_traj, axis=0)
        return Tp1_x
