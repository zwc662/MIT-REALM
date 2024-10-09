import multiprocessing as mp
import time
from typing import Callable, NamedTuple

import casadi as cs
import horizon.problem as prb
import ipdb
import jax.tree_util as jtu
import numpy as np
from attrs import define
from horizon.misc_function import shift_array
from horizon.transcriptions.transcriptor import Transcriptor
from horizon.variables import InputVariable, SingleParameter, StateVariable, Variable

from clfrl.dyn.dyn_types import BState, SControl, SState, State, STState
from clfrl.dyn.step_zoh import get_step_zoh
from clfrl.dyn_cs.task_cs import TaskCS
from clfrl.solvers.snopt_casadi import SnoptSolver
from clfrl.utils.jax_utils import tree_stack


def get_solver_opts(silent: bool = True, jit: bool = True, major_print_level: int = 0):
    jit_opts = {
        "jit": True,
        "compiler": "shell",
        "jit_options": {"flags": "-O3 -march=native -ffast-math -fopenmp", "verbose": False},
    }

    opts = {
        "print_time": 0,
        # "expand": True,
        "snopt": {
            "Major print level": major_print_level,
            "Minor print level": 0,
            "Major optimality tolerance": 1e-4,
            "System information": 0,
            "Summary frequency": 0,
            # "Summary file": 0,
            # "Print file": 9,
            "Timing level": 0,
        },
    }
    if jit:
        opts = opts | jit_opts
    if silent:
        opts["snopt"]["silence"] = True
    return opts


def set_bounds(var: Variable, lb: float, ub: float, scale: float = 1.0):
    var.setBounds(lb=[lb / scale], ub=[ub / scale])


@define
class MPCCfg:
    n_nodes: int
    dt: float
    cost_reg: float
    mpc_T: int
    transcription: str = "multiple_shooting"
    buffer: float = 1e-3


class MPCResult(NamedTuple):
    S_x: SState
    S_u: SControl
    S_unom: SControl
    ST_x: STState
    t_solve: float


class MPCVariables(NamedTuple):
    x: StateVariable
    u: InputVariable
    u_nom: SingleParameter


class MPCState(NamedTuple):
    var: MPCVariables
    s: SnoptSolver
    step: Callable
    nom_pol: Callable
    cfg: MPCCfg


def run_mpc(mpc_state: MPCState, x0: State) -> MPCResult:
    x = x0

    S_x, S_u, S_unom = [x0], [], []
    ST_x = []

    # Reset initial guesses.
    state_zeros = np.zeros_like(mpc_state.var.x.getLowerBounds())
    control_zeros = np.zeros_like(mpc_state.var.u.getLowerBounds())
    mpc_state.var.x.setInitialGuess(state_zeros)
    mpc_state.var.u.setInitialGuess(control_zeros)

    t0 = time.perf_counter()
    for ii in range(mpc_state.cfg.mpc_T):
        # print("[{:2}] MPC {:3}/{:3}".format(worker_id, ii + 1, cfg.mpc_T))
        mpc_state.var.x.setBounds(lb=x, ub=x, nodes=0)
        u_nom = mpc_state.nom_pol(x)
        mpc_state.var.u_nom.assign(u_nom)
        mpc_state.s.solve()
        sol = mpc_state.s.getSolutionDict()
        state, control = sol["state"], sol["control"]
        T_x, T_u = state.T, control.T

        S_u.append(T_u[0])
        S_unom.append(u_nom)
        ST_x.append(T_x)

        # Update initial guess.
        state_ig = shift_array(sol["state"], -1, 0.0)
        control_ig = shift_array(sol["control"], -1, 0.0)
        mpc_state.var.x.setInitialGuess(state_ig)
        mpc_state.var.u.setInitialGuess(control_ig)

        x = mpc_state.step(x, T_u[0])
        S_x.append(x)
    t1 = time.perf_counter()
    S_x, S_u, ST_x, S_unom = np.stack(S_x), np.stack(S_u), np.stack(ST_x), np.stack(S_unom)
    t_solve = t1 - t0

    return MPCResult(S_x, S_u, S_unom, ST_x, t_solve)


def mpc_sim_single(task: TaskCS, x0: BState, pol, cfg: MPCCfg) -> MPCResult:
    mpc_state = mpc_state_init(task, pol, cfg)
    return run_mpc(mpc_state, x0)


def mpc_sim(worker_id: int, task: TaskCS, b_x0: BState, pol, cfg: MPCCfg) -> MPCResult:
    prob = prb.Problem(cfg.n_nodes)
    prob.setDt(cfg.dt)

    state_, control_ = task.get_variables(prob)
    prob.setDynamics(task.xdot(state_, control_))

    Transcriptor.make_method(cfg.transcription, prob)
    task.add_constraints(state_, control_, prob, buffer=cfg.buffer)

    u_nom_ = prob.createSingleParameter("u_nom", task.nu)

    cost = cs.dot(control_, control_)
    prob.createIntermediateCost("cost", cfg.cost_reg * cost)

    nom_cost = cs.dot(control_ - u_nom_, control_ - u_nom_)
    prob.createIntermediateCost("nom_cost", nom_cost, 0)

    opts = get_solver_opts()
    s = SnoptSolver(prob, opts)
    step = get_step_zoh(task.task, cfg.dt)

    var = MPCVariables(state_, control_, u_nom_)

    data = []
    for ii, x0 in enumerate(b_x0):
        print("[{:2}] Solving {:4}/{:4}".format(worker_id, ii + 1, len(b_x0)))
        data.append(run_mpc(worker_id, var, s, step, pol, x0, cfg))

    data = tree_stack(data)
    return data


def mpc_state_init(task: TaskCS, pol, cfg: MPCCfg, solver_name: str = "solver") -> MPCState:
    prob = prb.Problem(cfg.n_nodes)
    prob.setDt(cfg.dt)

    state_, control_ = task.get_variables(prob)
    prob.setDynamics(task.xdot(state_, control_))

    Transcriptor.make_method(cfg.transcription, prob)
    task.add_constraints(state_, control_, prob, buffer=cfg.buffer)

    u_nom_ = prob.createSingleParameter("u_nom", task.nu)

    cost = cs.dot(control_, control_)
    prob.createIntermediateCost("cost", cfg.cost_reg * cost)

    opts = get_solver_opts()
    s = SnoptSolver(prob, opts, solver_name=solver_name)
    ipdb.set_trace()
    step = get_step_zoh(task.task, cfg.dt)

    var = MPCVariables(state_, control_, u_nom_)

    return MPCState(var, s, step, pol, cfg)


def mpc_worker_init(task: TaskCS, pol, cfg: MPCCfg, worker_state: dict):
    worker_state["mpc_state"] = mpc_state_init(task, pol, cfg)


def mpc_sim_worker(queue: mp.Queue, wid: int, task: TaskCS, b_x0: BState, pol, cfg: MPCCfg):
    solver_name = "snopt_{}".format(wid)
    mpc_state = mpc_state_init(task, pol, cfg, solver_name=solver_name)
    data = []
    n_completed_since_push = 0
    for ii, x0 in enumerate(b_x0):
        data.append(run_mpc(mpc_state, x0))
        n_completed_since_push += 1

        if n_completed_since_push % 10 == 0:
            queue.put((wid, n_completed_since_push))
            n_completed_since_push = 0
    queue.put((wid, n_completed_since_push))
    data = tree_stack(data)
    return data
