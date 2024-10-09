import casadi as cs
import horizon.problem as prb
from horizon import variables as sv

from clfrl.dyn.doubleint_wall import DoubleIntWall
from clfrl.dyn.dyn_types import State
from clfrl.dyn_cs.task_cs import TaskCS


class DoubleIntCS(TaskCS):
    TaskCls = DoubleIntWall

    def __init__(self):
        self._task = DoubleIntWall()

    @property
    def task(self) -> DoubleIntWall:
        return self._task

    def xdot(self, state: sv.StateVariable, control: sv.InputVariable):
        p, v = state[0], state[1]
        a = control[0]

        return cs.vertcat(v, a)

    def add_constraints(self, state: sv.StateVariable, control: sv.InputVariable, prob: prb.Problem, buffer: float):
        p, v = state[0], state[1]
        a = control[0]

        p.setBounds(lb=[-1.0 + buffer], ub=[1.0 - buffer])

        # prob.addConstraint("v_lb", v, lb=-1.0)
        # prob.addConstraint("v_ub", v, ub=1.0)
        #
        # prob.addConstraint("p_lb", p, lb=-1.0)
        # prob.addConstraint("p_ub", p, ub=1.0)

    def nominal_val_state(self) -> State:
        return self._task.nominal_val_state()
