import casadi as cs
import horizon.problem as prb
import horizon.utils.plotter as plotter
import ipdb
import matplotlib.pyplot as plt
import numpy as np
from horizon.misc_function import shift_array
from horizon.solvers import Solver
from horizon.transcriptions.transcriptor import Transcriptor
from horizon.utils.recedingHandler import RecedingHandler


def main():
    n_nodes = 25
    dt = 0.1
    mu = 0.2
    grav = 9.81
    prob = prb.Problem(n_nodes, receding=True)

    p = prob.createStateVariable("pos", dim=2)

    v = prob.createStateVariable("vel", dim=2)
    F = prob.createInputVariable("force", dim=2)

    p_tgt = prob.createParameter("pos_goal", dim=2)
    state = prob.getState()
    state_prev = state.getVarOffset(-1)
    x = state.getVars()

    xdot = cs.vertcat(v, F)  # - mu*grav*np.sign(v)
    prob.setDynamics(xdot)
    prob.setDt(dt)

    th = Transcriptor.make_method("multiple_shooting", prob)

    # set initial state (rest in zero)
    p.setBounds(lb=[0, 0], ub=[0, 0], nodes=0)
    v.setBounds(lb=[0, 0], ub=[0, 0], nodes=0)

    # final constraint
    # p.setBounds(lb=[1, 1], ub=[1, 1], nodes=n_nodes)
    goal_cnsrt = prob.createFinalConstraint("goal", p - p_tgt)
    v.setBounds(lb=[0, 0], ub=[0, 0], nodes=n_nodes)

    obs_center = np.array([0.7, 0.7])
    obs_r = 0.1
    obs = cs.sumsqr(p - obs_center) - obs_r**2

    obs_cnsrt = prob.createIntermediateConstraint("obstacle", obs, nodes=[])
    # obs_cnsrt = prob.createIntermediateConstraint('obstacle', obs)
    # intermediate cost ( i want to minimize the force! )
    prob.createIntermediateCost("cost", cs.sumsqr(F))

    traj = np.array([])
    opts = {"print_time": 0, "ipopt": {"print_level": 0}}
    solver = Solver.make_solver("ipopt", prob, opts)
    p_tgt.assign([1, 1])

    plt.ion()
    fig, ax = plt.subplots()
    ax.set_title("xy plane")
    ax.plot([0, 0], [0, 0], "bo", markersize=12)
    ax.plot([1, 1], [1, 1], "g*", markersize=12)
    (line_traj,) = ax.plot(0, 0)

    rec_nodes = -1
    for i in range(25):
        print(f"========== iteration {i} ==============")
        solver.solve()
        solution = solver.getSolutionDict()

        # update initial guess
        p_ig = shift_array(solution["pos"], -1, 0.0)
        v_ig = shift_array(solution["vel"], -1, 0.0)
        f_ig = shift_array(solution["force"], -1, 0.0)
        p.setInitialGuess(p_ig)
        v.setInitialGuess(v_ig)
        F.setInitialGuess(f_ig)

        # required bounds for setting intial position
        p.setBounds(solution["pos"][:, 1], solution["pos"][:, 1], 0)
        goal_cnsrt.setLowerBounds([-0.1, -0.1])
        goal_cnsrt.setUpperBounds([0.1, 0.1])

        # shift goal_cnsrt in horizon
        # goal_cnsrt.shift()
        # exit()
        # print(goal_cnsrt.getNodes())
        # print(goal_cnsrt.getBounds())
        goal_cnsrt.setNodes(goal_cnsrt.getNodes() - 1)
        # print(goal_cnsrt.getNodes())
        # print(goal_cnsrt.getBounds())
        # exit()

        # shift v bounds in horizon
        shifted_v_lb = shift_array(v.getLowerBounds(), -1, 0.0)
        shifted_v_ub = shift_array(v.getUpperBounds(), -1, 0.0)
        v.setBounds(shifted_v_lb, shifted_v_ub)

        traj = (
            np.hstack((traj, np.atleast_2d(solution["pos"][:, 0]).T))
            if traj.size
            else np.atleast_2d(solution["pos"][:, 0]).T
        )

        line_traj.set_xdata(traj[0])
        line_traj.set_ydata(traj[1])
        fig.canvas.draw()
        fig.canvas.flush_events()
        if i > 10:
            obs_cnsrt.setLowerBounds([0.0], nodes=np.arange(n_nodes))
            circle = plt.Circle(obs_center, radius=obs_r, fc="r")
            ax.add_patch(circle)

        rec_nodes = rec_nodes - 1


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
