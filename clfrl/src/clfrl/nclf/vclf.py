from typing import NamedTuple

import jax
import jax.lax as lax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from attrs import define
from flax import struct
from loguru import logger

import clfrl.utils.typed_ft as ft
from clfrl.dyn.dyn_types import BBControl, BBState, BState, BTState, State
from clfrl.dyn.task import Task
from clfrl.nclf.min_norm_control import min_norm_control_fixed
from clfrl.nclf.sim_nclf import SimNCLF
from clfrl.networks.mlp import MLP
from clfrl.networks.network_utils import HidSizes, get_act_from_str
from clfrl.networks.optim import get_default_tx
from clfrl.networks.train_state import TrainState
from clfrl.networks.vclf_net import VecFn
from clfrl.utils.grad_utils import compute_norm
from clfrl.utils.jax_types import BBFloat, MetricsDict
from clfrl.utils.jax_utils import jax_vmap, rep_vmap, swaplast
from clfrl.utils.loss_utils import weighted_sum_dict
from clfrl.utils.rng import PRNGKey
from clfrl.utils.schedules import Schedule


@define
class VCLFTrainCfg:
    # Batch size to use when updating.
    batch_size: int
    desc_rate: float

    goal_zero_margin: float = 1e-3
    stop_u: bool = False


@define
class VCLFEvalCfg:
    # How long to rollout during eval.
    eval_rollout_T: int


@define
class VCLFCfg:
    act: str
    lr: Schedule

    hids: HidSizes
    train_cfg: VCLFTrainCfg
    eval_cfg: VCLFEvalCfg

    @property
    def alg_name(self) -> str:
        return "VCLF"


class VCLF(struct.PyTreeNode):
    collect_idx: int
    key: PRNGKey
    Vx: TrainState[State]

    cfg: VCLFCfg = struct.field(pytree_node=False)
    task: Task = struct.field(pytree_node=False)

    class EvalData(NamedTuple):
        bT_x_plot: BTState

        bb_Xs: BBFloat
        bb_Ys: BBFloat
        bb_Vx: BBState
        bb_Vdot: BBFloat
        bb_div: BBFloat
        bb_curlfro: BBFloat
        bb_u: BBControl
        info: MetricsDict

    @classmethod
    def create(cls, seed: int, task: Task, cfg: VCLFCfg) -> "VCLF":
        key, key_V = jr.split(jr.PRNGKey(seed), 2)
        pol_obs, V_obs = np.zeros(task.n_polobs), np.zeros(task.n_Vobs)
        act = get_act_from_str(cfg.act)

        goal_pt = task.goal_pt()
        goal_Vobs, _ = task.get_obs(goal_pt)

        # Define Vx network.
        Vx_cls = ft.partial(MLP, cfg.hids, act)
        Vx_def = VecFn(Vx_cls, task.nx)
        Vx_tx = get_default_tx(cfg.lr.make())
        Vx = TrainState.create_from_def(key_V, Vx_def, (pol_obs,), Vx_tx)

        return VCLF(jnp.array(0, dtype=jnp.int32), key, Vx, cfg, task)

    @property
    def train_cfg(self):
        return self.cfg.train_cfg

    @property
    def desc_rate(self):
        return self.train_cfg.desc_rate

    @ft.partial(jax.jit, donate_argnums=0)
    def sample_batch(self):
        key = jr.fold_in(self.key, self.collect_idx)
        b_x0 = self.task.sample_train_x0(key, self.train_cfg.batch_size)
        return self.replace(collect_idx=self.collect_idx + 1), b_x0

    def get_Vx(self, params, state: State):
        Vx_obs, _ = self.task.get_obs(state)
        return self.Vx.apply_fn(params, Vx_obs)

    @ft.partial(jax.jit, donate_argnums=0)
    def update(self, b_x: BState, loss_weights: MetricsDict):
        def loss_fn(params):
            Vx_apply = ft.partial(self.get_Vx, params)

            # 1: Vx(0) = 0.
            Vx_goal = Vx_apply(goal_pt)
            loss_goal = jnp.sum(Vx_goal**2)

            # 2: tr( Vxx(0) ) > 0. Eigenvalues of V should be positive.
            Vxx_goal = jax.jacobian(Vx_apply)(goal_pt)
            V_goal_div = jnp.trace(Vxx_goal)
            loss_goal_div = jnn.relu(-V_goal_div) ** 2

            # 3: Compute min norm controller.
            b_Vx = jax_vmap(Vx_apply)(b_x)
            b_f = jax_vmap(self.task.f)(b_x)
            b_G = jax_vmap(self.task.G)(b_x)
            b_u, _, _ = jax_vmap(ft.partial(min_norm_control_fixed, self.desc_rate, u_lb, u_ub))(b_Vx, b_f, b_G)

            if self.train_cfg.stop_u:
                logger.info("No gradient through u!")
                b_u = lax.stop_gradient(b_u)

            # 3: cts time V descent. Vdot + desc_rate <= 0
            b_xdot = b_f + jnp.sum(b_G * b_u[:, None, :], axis=-1)
            b_Vdot = jnp.sum(b_Vx * b_xdot, axis=-1)
            b_Vdot_constr = b_Vdot + self.desc_rate
            loss_Vdesc = jnp.mean(b_Vdot_constr**2)
            acc_Vdesc = jnp.mean(b_Vdot_constr < 0)

            # 4: Zero curl. Penalize Frobenius norm of Vxx - Vxx^T
            b_Vxx = jax_vmap(jax.jacobian(Vx_apply))(b_x)
            b_curl = jnp.sum((b_Vxx - swaplast(b_Vxx)) ** 2, axis=(-1, -2))
            loss_curl = jnp.mean(b_curl)

            loss_dict = {
                "Loss/Goal": loss_goal,
                "Loss/Goal Divergence": loss_goal_div,
                "Loss/V_desc": loss_Vdesc,
                "Loss/Curl": loss_curl,
            }
            info_dict = {"Acc/V_desc": acc_Vdesc}
            loss = weighted_sum_dict(loss_dict, loss_weights)
            return loss, loss_dict | info_dict

        u_lb, u_ub = np.array([-1.0]), np.array([+1.0])
        goal_pt = self.task.goal_pt()

        grads, info = jax.grad(loss_fn, has_aux=True)(self.Vx.params)
        info["Qh_grad"] = compute_norm(grads)
        Vx = self.Vx.apply_gradients(grads)
        return self.replace(Vx=Vx), info

    # def update_weights(self, b_x: BState, loss_weights: MetricsDict) -> tuple[MetricsDict, MetricsDict]:
    #     goal_pt = self.task.goal_pt()
    #     Vx_goal = self.get_Vx(self.Vx.params, goal_pt)
    #     V_other = jax_vmap(ft.partial(self.get_V, self.V.params))(b_x)
    #     V_other_min = V_other.min()
    #
    #     # If V_goal is not smaller than all other points by MARGIN, then increase the weight.
    #     # Otherwise, decay the weight slowly.
    #     MARGIN = 1e-2
    #     V_goal_smallest = V_goal + self.train_cfg.goal_zero_margin < V_other_min
    #     # Positive is good.
    #     V_goal_diff = V_other_min - V_goal
    #
    #     factor_decr = 0.999
    #     factor_incr = 1.1
    #     goal_nonzero_old = loss_weights["Loss/Nonzero"]
    #     goal_nonzero = jnp.where(V_goal_smallest, factor_decr, factor_incr) * goal_nonzero_old
    #
    #     info = {"weights/V_goal_diff": V_goal_diff, "weights/Nonzero": goal_nonzero}
    #     return loss_weights | {"Loss/Nonzero": goal_nonzero}, info

    def get_control(self, state: State):
        u_lb, u_ub = np.array([-1.0]), np.array([+1.0])
        Vx_apply = ft.partial(self.get_Vx, self.Vx.params)

        Vx = Vx_apply(state)
        f = self.task.f(state)
        G = self.task.G(state)
        u, _, _ = min_norm_control_fixed(self.desc_rate, u_lb, u_ub, Vx, f, G)
        return self.task.chk_u(u)

    @jax.jit
    def eval(self) -> EvalData:
        # Get states for plotting and for metrics.
        b_x0_plot = self.task.get_plot_x0()
        b_x0_metric = self.task.get_metric_x0()
        b_x0_rng = self.task.get_plot_rng_x0()

        # Rollout using the min norm controller.
        sim = SimNCLF(self.task, self.get_control, self.cfg.eval_cfg.eval_rollout_T)
        bT_x_plot, _, _ = jax_vmap(sim.rollout_plot)(b_x0_plot)
        bT_x_metric, _ = jax_vmap(sim.rollout)(b_x0_metric)

        Vx_apply = ft.partial(self.get_Vx, self.Vx.params)

        def get_Vx_info(state):
            Vx = Vx_apply(state)
            Vxx = jax.jacobian(Vx_apply)(state)
            div = jnp.trace(Vxx)
            curlfro = jnp.sum((Vxx - Vxx.T) ** 2)
            f, G = self.task.f(state), self.task.G(state)
            u, _, _ = min_norm_control_fixed(self.desc_rate, u_lb, u_ub, Vx, f, G)
            xdot = f + jnp.sum(G * u, axis=-1)
            Vdot = jnp.dot(xdot, Vx)
            return Vx, Vdot, div, curlfro, u

        u_lb, u_ub = np.array([-1]), np.array([+1])
        bb_x, bb_Xs, bb_Ys = self.task.get_contour_x0()
        bb_Vx, bb_Vdot, bb_div, bb_curlfro, bb_u = rep_vmap(get_Vx_info, rep=2)(bb_x)

        bTl_ls = rep_vmap(self.task.l_components, rep=2)(bT_x_metric)

        l_mean = bTl_ls.sum(1).mean(0)
        l_final = bTl_ls[:, -1, :].mean(0)
        eval_info = {"Cost Mean": l_mean, "Cost Final Mean": l_final}

        return self.EvalData(bT_x_plot, bb_Xs, bb_Ys, bb_Vx, bb_Vdot, bb_div, bb_curlfro, bb_u, eval_info)
