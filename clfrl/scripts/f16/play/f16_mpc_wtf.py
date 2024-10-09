import casadi as ca
import casadi_f16.f16
import horizon.problem as prb
import ipdb
import matplotlib.pyplot as plt
import numpy as np
from horizon.transcriptions.transcriptor import Transcriptor
from loguru import logger

from clfrl.dyn.f16_gcas import F16GCAS
from clfrl.dyn.sim_cts_pbar import SimCtsPbar
from clfrl.dyn_cs.f16gcas_cs import F16GCASCS
from clfrl.mpc.mpc import get_solver_opts
from clfrl.solvers.snopt_casadi import SnoptSolver
from clfrl.utils.jax_utils import jax2np, jax_default_x32, jax_jit, jax_use_cpu, jax_vmap
from clfrl.utils.logging import set_logger_format
from clfrl.utils.paths import get_script_plot_dir


def main():
    plot_dir = get_script_plot_dir()
    jax_use_cpu()
    jax_default_x32()
    set_logger_format()
    task = F16GCAS()
    task_ca = F16GCASCS()

    dt = task.dt

    x0 = task.nominal_val_state()
    x0[task.H] = 600
    x0[task.THETA] = 0.2

    prob = prb.Problem(1, abstract_casadi_type=ca.MX)
    prob.setDt(dt)

    state_, control_ = task_ca.get_variables(prob, mx=True)
    prob.setDynamics(task_ca.xdot(state_, control_))

    Transcriptor.make_method("multiple_shooting", prob)
    task_ca.add_constraints(state_, control_, prob, buffer=1e-4)

    prob.setInitialState(x0)
    state_.setInitialGuess(x0)

    # Minimize cost.
    cost = ca.sumsqr(control_)
    prob.createIntermediateCost("cost", cost)

    opts = get_solver_opts(silent=False, jit=False, major_print_level=1)
    s = SnoptSolver(prob, opts, solver_name="snopt")
    logger.info("Solving...")
    s.solve()
    logger.info("Solving... Done!")

    sol = s.getSolutionDict()
    x, u = sol["state"], sol["control"]

    print(x.shape, u.shape)
    solver_inp = ca.veccat(x, u)
    print("solver_inp.shape: ", solver_inp.shape)

    xx, gg = s.prob_dict["x"], s.prob_dict["g"]
    jac = ca.jacobian(gg, xx)
    jac_fn = ca.Function("jac_fn", [xx], [jac])

    # out = np.array(jac_fn(solver_inp))
    # print(out)

    # ipdb.set_trace()


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
