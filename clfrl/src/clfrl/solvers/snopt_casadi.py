from typing import Dict

import casadi as cs
from horizon.functions import Cost, RecedingCost, RecedingResidual, Residual
from horizon.problem import Problem
from horizon.solvers import Solver

from clfrl.utils.hidden_print import HiddenPrints


class NlpsolSolver2(Solver):
    def __init__(self, prb: Problem, opts: Dict, solver_plugin: str, solver_name: str = "solver") -> None:

        super().__init__(prb, opts=opts)

        # generate problem to be solved
        self.var_container = self.prb.var_container
        self.fun_container = self.prb.function_container

        self.vars_impl = dict()
        self.pars_impl = dict()

        self.cond_warm_start = self.opts.get("ipopt.warm_start_init_point", "no") == "yes"
        self.lam_x0 = None
        self.lam_g0 = None

        # dictionary of implemented variables
        self.dict_sol = dict(x0=None, lbx=None, ubx=None, lbg=None, ubg=None, p=None)

        j, w, g, p = self.build()
        # implement the abstract state variable with the current node
        # self.prb.var_container.build()
        # implement the constraints and the cost functions with the current node
        # self.function_container.build()

        # get j, w, g
        # j = self.function_container.getCostFImplSum()
        # w = self.var_container.getVarImplList()
        # g = self.function_container.getCnstrFList()
        # p = self.var_container.getParameterList()

        self.prob_dict = {"f": j, "x": w, "g": g, "p": p}

        # create solver from prob
        self.solver = cs.nlpsol(solver_name, solver_plugin, self.prob_dict, self.opts)

    def build(self):
        """
        fill the dictionary "state_var_impl"
            - key: nodes (nNone, n0, n1, ...) nNone contains single variables that are not projected in nodes
            - val: dict with name and value of implemented variable
        """

        # todo it seems tht i only need self.vars in var_container.
        # ORDERED AS VARIABLES
        # build variables
        var_list = list()
        for var in self.var_container.getVarList(offset=False):
            # x_2_3 --> dim 2 of node 3
            # order is: x_0_0, x_1_0, x_0_1, x_1_1 ...
            var_list.append(var.getImpl())
        w = cs.veccat(*var_list)

        # build parameters
        par_list = list()
        for par in self.var_container.getParList(offset=False):
            par_list.append(par.getImpl())
        p = cs.veccat(*par_list)

        # build constraint functions list
        fun_list = list()
        for fun in self.fun_container.getCnstr().values():
            fun_to_append = fun.getImpl()
            if fun_to_append is not None:
                fun_list.append(fun_to_append)
        g = cs.veccat(*fun_list)

        # todo: residual, recedingResidual should be the same class
        # treat differently cost and residual (residual must be quadratized)
        fun_list = list()
        for fun in self.fun_container.getCost().values():
            fun_to_append = fun.getImpl()
            if fun_to_append is not None:
                if type(fun) in (Cost, RecedingCost):
                    fun_list.append(fun_to_append[:])
                elif type(fun) in (Residual, RecedingResidual):
                    fun_list.append(cs.sumsqr(fun_to_append[:]))
                else:
                    raise Exception("wrong type of function found in fun_container")

        # if it is empty, just set j to []
        j = cs.sum1(cs.veccat(*fun_list)) if fun_list else []

        return j, w, g, p

    def solve(self) -> bool:

        # update lower/upper bounds of variables
        lbw = self._getVarList("lb")
        ubw = self._getVarList("ub")
        # update initial guess of variables
        w0 = self._getVarList("ig")
        # update parameters
        p = self._getParList()
        # update lower/upper bounds of constraints
        lbg = self._getFunList("lb")
        ubg = self._getFunList("ub")

        # last guard
        if lbg.shape != self.prob_dict["g"].shape:
            raise ValueError(
                f'Constraint bounds have mismatching shape: {lbg.shape}. Allowed dimensions: {self.prob_dict["g"].shape}. '
                f"Be careful: if you added constraints or variables after loading the problem, you have to rebuild it before solving it!"
            )

        # update solver arguments
        self.dict_sol["x0"] = w0
        self.dict_sol["lbx"] = lbw
        self.dict_sol["ubx"] = ubw
        self.dict_sol["lbg"] = lbg
        self.dict_sol["ubg"] = ubg
        self.dict_sol["p"] = p

        # solve
        sol = self.solver(**self.dict_sol)

        if self.cond_warm_start:
            self.dict_sol["lam_x0"] = sol["lam_x"]
            self.dict_sol["lam_g0"] = sol["lam_g"]

        self.cnstr_solution = self._createCnsrtSolDict(sol)

        # retrieve state and input trajector

        # get solution dict
        self.var_solution = self._createVarSolDict(sol)

        # get solution as state/input
        self._createVarSolAsInOut(sol)
        self.var_solution["x_opt"] = self.x_opt
        self.var_solution["u_opt"] = self.u_opt

        # build dt_solution as an array
        self._createDtSol()

        return self.solver.stats()["success"]

    def getSolutionDict(self):
        return self.var_solution

    def getConstraintSolutionDict(self):
        return self.cnstr_solution

    def getDt(self):
        return self.dt_solution

    def getSolutionState(self):
        return self.var_solution["x_opt"]

    def getSolutionInput(self):
        return self.var_solution["u_opt"]


class SnoptSolver(NlpsolSolver2):
    def __init__(self, prb: Problem, opts: Dict, solver_name: str = "solver") -> None:
        # remove ilqr options from solver options
        filtered_opts = None
        if opts is not None:
            filtered_opts = {k: opts[k] for k in opts.keys() if not k.startswith("ilqr.")}
        super().__init__(prb, opts=filtered_opts, solver_plugin="snopt", solver_name=solver_name)
