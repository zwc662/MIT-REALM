import functools as ft

import jax.lax as lax
import numpy as np
from diffrax import (
    Bosh3,
    ConstantStepSize,
    DirectAdjoint,
    ODETerm,
    PIDController,
    RecursiveCheckpointAdjoint,
    SaveAt,
    Tsit5,
    diffeqsolve,
)
from loguru import logger

from pncbf.dyn.dyn_types import State
from pncbf.dyn.odeint import tsit5, tsit5_dense
from pncbf.dyn.task import Task
from pncbf.utils.jax_utils import concat_at_end, jax_vmap, merge01
from pncbf.utils.none import get_or


def get_solver(solver: str):
    if solver == "bosh3":
        return Bosh3()
    if solver == "tsit5":
        return Tsit5()

    raise NotImplementedError("")


def sim_step(task: Task, policy, dt: float, x0: State, solver: str = "tsit5"):
    def body(t, state, args):
        control = policy(state)
        return task.xdot(state, control)

    term = ODETerm(body)
    solver = get_solver(solver)
    saveat = SaveAt(t1=True)
    stepsize_controller = ConstantStepSize()
    solution = diffeqsolve(
        term,
        solver,
        t0=0,
        t1=dt,
        dt0=dt / 2,
        y0=x0,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=4,
    )
    assert solution.ys.shape == (1, task.nx)
    return solution.ys[0]


class SimCtsReal:
    def __init__(
        self,
        task: Task,
        policy,
        tf: float,
        result_dt: float,
        dt0: float = None,
        use_obs: bool = False,
        max_steps: int = 256,
        use_pid: bool = True,
        solver: str = "tsit5",
    ):
        self.task = task
        self.policy = policy
        self.result_dt = result_dt
        self.tf = tf
        self.use_obs = use_obs
        self.max_steps = max_steps
        self.use_pid = use_pid
        self.dt0 = get_or(dt0, result_dt)
        self.solver = solver

    def get_control(self, state):
        if self.use_obs:
            Vobs, polobs = self.task.get_obs(state)
            return self.policy(polobs)

        return self.policy(state)

    def rollout_plot(self, x0: State):
        def body(t, state, args):
            control = self.get_control(state)
            return self.task.xdot(state, control)

        term = ODETerm(body)
        solver = get_solver(self.solver)

        T = int(round(self.tf / self.result_dt))
        T_ts = np.linspace(0, self.tf, num=T + 1)
        saveat = SaveAt(ts=T_ts)
        # saveat = SaveAt(dense=True)
        # adjoint = DirectAdjoint()
        adjoint = RecursiveCheckpointAdjoint()
        if self.use_pid:
            stepsize_controller = PIDController(pcoeff=0.1, icoeff=0.4, rtol=1e-5, atol=1e-5)
        else:
            stepsize_controller = ConstantStepSize()
        solution = diffeqsolve(
            term,
            solver,
            t0=0,
            t1=self.tf,
            dt0=self.dt0,
            y0=x0,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            adjoint=adjoint,
            max_steps=self.max_steps,
        )
        T_states = solution.ys
        return T_states, solution.ts, solution.stats


class SimCts:
    def __init__(self, task: Task, policy, T: int, interp_pts: int, pol_dt: float, use_obs: bool = False):
        self.task = task
        self.policy = policy
        self.T = T
        self.interp_pts = interp_pts
        self.pol_dt = pol_dt
        self.use_obs = use_obs

    def get_control(self, state):
        if self.use_obs:
            Vobs, polobs = self.task.get_obs(state)
            return self.policy(polobs)

        return self.policy(state)

    def rollout_plot(self, x0: State):
        def body_zoh(t_state: tuple[float, State], _):
            t, state = t_state
            control = self.get_control(state)
            xdot_with_u = ft.partial(self.task.xdot, control=control)
            states_interp = tsit5_dense(self.pol_dt, 5, xdot_with_u, state)
            ts = np.linspace(0, self.pol_dt, num=self.interp_pts + 1)
            T_states = jax_vmap(states_interp.evaluate)(ts)
            state_new = T_states[-1]
            return (t + ts[-1], state_new), (T_states[:-1], control, t + ts[:-1])

        (t_f, state_f), (TT_state, T_control, TT_ts) = lax.scan(body_zoh, (0.0, x0), None, self.T)
        T_state, T_ts = merge01(TT_state), merge01(TT_ts)
        Tp1_state = concat_at_end(T_state, state_f, axis=0)
        T_ts = concat_at_end(T_ts, t_f, axis=0)

        return Tp1_state, T_control, T_ts


def integrate(interp_pts: int, dt: float, xdot, x0, T_u):
    def body_zoh(t_state: tuple[float, State], control):
        t, state = t_state
        xdot_with_u = ft.partial(xdot, control=control)
        states_interp = tsit5_dense(dt, 5, xdot_with_u, state)
        ts = np.linspace(0, dt, num=interp_pts + 1)
        T_states = jax_vmap(states_interp.evaluate)(ts)
        state_new = T_states[-1]
        return (t + ts[-1], state_new), (T_states[:-1], t + ts[:-1])

    (t_f, state_f), (TT_state, TT_ts) = lax.scan(body_zoh, (0.0, x0), T_u, len(T_u))
    T_state, T_ts = merge01(TT_state), merge01(TT_ts)
    Tp1_state = concat_at_end(T_state, state_f, axis=0)
    Tp1_ts = concat_at_end(T_ts, t_f, axis=0)

    return Tp1_state, Tp1_ts
