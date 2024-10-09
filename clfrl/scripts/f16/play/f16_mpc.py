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
from clfrl.plotting.plot_utils import plot_boundaries
from clfrl.solvers.snopt_casadi import SnoptSolver
from clfrl.utils.jax_utils import jax2np, jax_default_x32, jax_jit, jax_use_cpu, jax_vmap
from clfrl.utils.logging import set_logger_format
from clfrl.utils.paths import get_script_plot_dir


def main():
    # See if jaxf16 and casadif16 agree on a trajectory using PID.
    plot_dir = get_script_plot_dir()
    jax_use_cpu()
    jax_default_x32()
    set_logger_format()
    task = F16GCAS()
    task_ca = F16GCASCS()

    # tf = 10.0
    tf = 5.0
    dt = task.dt
    n_steps = int(round(tf / dt))

    x0 = task.nominal_val_state()
    x0[task.H] = 600
    # x0[task.THETA] = 0.2

    pol = task.nom_pol_pid

    # #################################################################
    # x_ = ca.MX.sym("x", task.nx)
    # u_ = ca.MX.sym("u", task.nu)
    # xdot_ = task_ca.xdot(x_, u_)
    # jac_x_ = ca.jacobian(xdot_, x_)
    # jac_u_ = ca.jacobian(xdot_, u_)
    # jac_fn = ca.Function("jac_fn", [x_, u_], [jac_x_, jac_u_])
    #
    # u0 = pol(x0)
    # jx0, ju0 = jac_fn(x0, u0)
    # ca.fmax()
    # ipdb.set_trace()
    # #################################################################

    sim = SimCtsPbar(task, pol, n_steps, dt, dt0=dt, use_obs=False, use_pid=False, max_steps=n_steps, solver="bosh3")
    T_x_nom, T_t_nom = jax2np(jax_jit(sim.rollout_plot)(x0))
    T_u_nom = jax_vmap(pol)(T_x_nom)

    print("dt: {}, {}".format(dt, np.diff(T_t_nom).mean()))

    #################################
    # Try manual integrating.
    T_x_nom2 = task_ca.integrate_rk4(x0, T_u_nom[:-1, :], dt)
    assert T_x_nom2.shape == (n_steps + 1, task.nx)
    #################################

    # Now, try the same with casadi and snopt.
    prob = prb.Problem(n_steps, abstract_casadi_type=ca.MX)
    prob.setDt(dt)

    state_, control_ = task_ca.get_variables(prob, mx=True)
    prob.setDynamics(task_ca.xdot(state_, control_, scale=True))

    Transcriptor.make_method("multiple_shooting", prob)
    task_ca.add_constraints(state_, control_, prob, buffer=1e-4)

    # Try and match the controls.
    for kk in range(n_steps):
        cost = ca.sumsqr(control_ - T_u_nom[kk])
        prob.createIntermediateCost(f"cost{kk}", cost, kk)

    # Set initial state.
    prob.setInitialState(x0 / task_ca.x_scale)
    state_.setInitialGuess(x0 / task_ca.x_scale, 0)
    for kk in range(n_steps):
        control_.setBounds(lb=T_u_nom[kk, :], ub=T_u_nom[kk, :], nodes=kk)
        control_.setInitialGuess(T_u_nom[kk, :], kk)
        state_.setInitialGuess(T_x_nom[kk + 1, :] / task_ca.x_scale, kk + 1)

    opts = get_solver_opts(silent=False, jit=False, major_print_level=1)
    s = SnoptSolver(prob, opts, solver_name="snopt")

    for name, fun in s.fun_container.getCnstr().items():
        print(f"{name}: {fun}")

    logger.info("Solving...")
    s.solve()
    logger.info("Solving... Done!")
    sol = s.getSolutionDict()
    T_x_mpc, T_u_mpc = np.array(sol["state"].T), sol["control"].T
    T_x_mpc = T_x_mpc * task_ca.x_scale

    # Plot a comparison.
    x_labels = task.x_labels
    figsize = np.array([5, 1.0 * task.nx])
    fig, axes = plt.subplots(task.nx, figsize=figsize, sharex=True, layout="constrained")
    for ii, ax in enumerate(axes):
        ax.plot(T_t_nom, T_x_nom[:, ii], color="C0", lw=1.0, label="Jax")
        ax.plot(T_t_nom, T_x_mpc[:, ii], color="C1", lw=1.0, label="Casadi")
        ax.plot(T_t_nom, T_x_nom2[:, ii], color="C2", lw=1.0, label="Manual")
        ax.set_ylabel(x_labels[ii], rotation=0, ha="right")
    axes[0].legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, 1.0), fontsize="x-small")
    # Plot constraint boundaries.
    task.plot_boundaries(axes)
    # Plot training boundaries.
    plot_boundaries(axes, task.train_bounds())
    fig.savefig(plot_dir / f"mpc_compare.pdf")
    plt.close(fig)

    ipdb.set_trace()


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
