from typing import NamedTuple

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from attrs import define
from flax import struct

import clfrl.utils.typed_ft as ft
from clfrl.dyn.dyn_types import BBControl, BState, BTState, Control, State
from clfrl.dyn.task import Task
from clfrl.nclf.sim_nclf import SimNCLF
from clfrl.networks.mlp import MLP
from clfrl.networks.nclf import NormSqMetricFn2, NormSqMetricFn
from clfrl.networks.network_utils import HidSizes, get_act_from_str
from clfrl.networks.optim import get_default_tx
from clfrl.networks.pol_det import PolDet
from clfrl.networks.train_state import TrainState
from clfrl.utils.grad_utils import compute_norm
from clfrl.utils.jax_types import BBFloat, FloatScalar, MetricsDict
from clfrl.utils.jax_utils import jax_vmap, rep_vmap
from clfrl.utils.loss_utils import weighted_sum_dict
from clfrl.utils.rng import PRNGKey
from clfrl.utils.schedules import Schedule


@define
class NCLFPolTrainCfg:
    # Batch size to use when updating.
    batch_size: int
    desc_lam: float
    desc_rate: float

    goal_zero_margin: float = 1e-3
    use_rate: bool = True


@define
class NCLFPolEvalCfg:
    # How long to rollout during eval.
    eval_rollout_T: int


@define
class NCLFPolCfg:
    act: str
    lr: Schedule

    hids: HidSizes
    pol_hids: HidSizes

    train_cfg: NCLFPolTrainCfg
    eval_cfg: NCLFPolEvalCfg

    @property
    def alg_name(self) -> str:
        return "NCLFPol"


class NCLFPol(struct.PyTreeNode):
    collect_idx: int
    key: PRNGKey
    V: TrainState[FloatScalar]
    pol: TrainState[Control]

    cfg: NCLFPolCfg = struct.field(pytree_node=False)
    task: Task = struct.field(pytree_node=False)

    class EvalData(NamedTuple):
        bT_x_plot: BTState

        bb_Xs: BBFloat
        bb_Ys: BBFloat
        bb_V: BBFloat
        bb_Vdot: BBFloat
        bb_u: BBControl
        info: MetricsDict

    @classmethod
    def create(cls, seed: int, task: Task, cfg: NCLFPolCfg) -> "NCLFPol":
        key, key_V, key_pol = jr.split(jr.PRNGKey(seed), 3)
        pol_obs, V_obs = np.zeros(task.n_polobs), np.zeros(task.n_Vobs)
        act = get_act_from_str(cfg.act)

        goal_pt = task.goal_pt()
        goal_Vobs, _ = task.get_obs(goal_pt)

        # Define V network.
        V_cls = ft.partial(MLP, cfg.hids, act)
        V_def = NormSqMetricFn(V_cls, goal_Vobs)
        # V_def = NormSqMetricFn2(V_cls, goal_Vobs)
        V_tx = get_default_tx(cfg.lr.make())
        V = TrainState.create_from_def(key_V, V_def, (pol_obs,), V_tx)

        # Define pol network.
        pol_base_cls = ft.partial(MLP, cfg.pol_hids, act)
        pol_def = PolDet(pol_base_cls, task.nu)
        pol_tx = get_default_tx(cfg.lr.make())
        pol = TrainState.create_from_def(key_pol, pol_def, (pol_obs,), pol_tx)

        return NCLFPol(jnp.array(0, dtype=jnp.int32), key, V, pol, cfg, task)

    @property
    def train_cfg(self):
        return self.cfg.train_cfg

    @property
    def desc_lam(self):
        return self.train_cfg.desc_lam

    @property
    def desc_rate(self):
        return self.train_cfg.desc_rate

    @ft.partial(jax.jit, donate_argnums=0)
    def sample_batch(self):
        key = jr.fold_in(self.key, self.collect_idx)
        b_x0 = self.task.sample_train_x0(key, self.train_cfg.batch_size)
        return self.replace(collect_idx=self.collect_idx + 1), b_x0

    def get_V(self, params, state: State):
        V_obs, _ = self.task.get_obs(state)
        return self.V.apply_fn(params, V_obs)

    def get_control(self, params, state: State):
        _, pol_obs = self.task.get_obs(state)
        return self.pol.apply_fn(params, pol_obs)

    @ft.partial(jax.jit, donate_argnums=0)
    def update(self, b_x: BState, loss_weights: MetricsDict):
        def loss_fn(V_params, pol_params):
            V_apply = ft.partial(self.get_V, V_params)
            pol_apply = ft.partial(self.get_control, pol_params)

            # 1: V(0) = 0.
            V_goal = V_apply(goal_pt)
            loss_goal = V_goal**2

            # 2: Vx(0) = 0    (because its the minimum)
            Vx_goal = jax.grad(V_apply)(goal_pt)
            loss_goal_grad = jnp.sum(Vx_goal**2)

            # 3: V(nonzero) > V(0). Penalize V(0) - V(nonzero) > 0.
            b_V = jax_vmap(V_apply)(b_x)
            V_mean = b_V.mean()
            loss_nonzero = jnp.mean(jnn.relu((V_goal - b_V) / V_mean + self.train_cfg.goal_zero_margin) ** 2)
            acc_nonzero = jnp.mean(V_goal < b_V)

            # 4: Compute min norm controller.
            b_Vx = jax_vmap(jax.grad(V_apply))(b_x)
            b_f = jax_vmap(self.task.f)(b_x)
            b_G = jax_vmap(self.task.G)(b_x)
            b_u = jax_vmap(pol_apply)(b_x)

            # 5: cts time V descent. Vdot + lambda * V <= 0
            b_xdot = b_f + jnp.sum(b_G * b_u[:, None, :], axis=-1)
            b_Vdot = jnp.sum(b_Vx * b_xdot, axis=-1)

            if self.train_cfg.use_rate:
                b_Vdot_constr = b_Vdot + self.desc_rate
                loss_Vdesc = jnp.mean(b_Vdot_constr**2)
            else:
                b_Vdot_constr = b_Vdot + self.desc_lam * b_V
                loss_Vdesc = jnp.mean(jnn.relu(b_Vdot_constr + Vdesc_eps) ** 2)
            acc_Vdesc = jnp.mean(b_Vdot < 0)

            loss_dict = {
                "Loss/Goal": loss_goal,
                "Loss/Goal Grad": loss_goal_grad,
                "Loss/V_desc": loss_Vdesc,
                "Loss/Nonzero": loss_nonzero,
            }
            info_dict = {"Acc/V_desc": acc_Vdesc, "Acc/Nonzero": acc_nonzero}
            loss = weighted_sum_dict(loss_dict, loss_weights)
            return loss, loss_dict | info_dict

        Vdesc_eps = 1e-2
        goal_pt = self.task.goal_pt()

        (V_grads, pol_grads), info = jax.grad(loss_fn, argnums=(0, 1), has_aux=True)(self.V.params, self.pol.params)
        info["Qh_grad"] = compute_norm(V_grads)
        info["pol_grad"] = compute_norm(pol_grads)
        V = self.V.apply_gradients(V_grads)
        pol = self.pol.apply_gradients(pol_grads)
        return self.replace(V=V, pol=pol), info

    @jax.jit
    def eval(self) -> EvalData:
        # Get states for plotting and for metrics.
        b_x0_plot = self.task.get_plot_x0()
        b_x0_metric = self.task.get_metric_x0()
        b_x0_rng = self.task.get_plot_rng_x0()

        # Rollout using the min norm controller.
        sim = SimNCLF(self.task, ft.partial(self.get_control, self.pol.params), self.cfg.eval_cfg.eval_rollout_T)

        bT_x_plot, _, _ = jax_vmap(sim.rollout_plot)(b_x0_plot)
        bT_x_metric, _ = jax_vmap(sim.rollout)(b_x0_metric)

        def get_V_info(state):
            V, Vx = jax.value_and_grad(ft.partial(self.get_V, self.V.params))(state)
            f, G = self.task.f(state), self.task.G(state)

            u = self.get_control(self.pol.params, state)
            xdot = f + jnp.sum(G * u, axis=-1)
            Vdot = jnp.dot(xdot, Vx)
            return V, Vdot, u

        bb_x, bb_Xs, bb_Ys = self.task.get_contour_x0()
        bb_V, bb_Vdot, bb_u = rep_vmap(get_V_info, rep=2)(bb_x)

        bTl_ls = rep_vmap(self.task.l_components, rep=2)(bT_x_metric)

        l_mean = bTl_ls.sum(1).mean(0)
        l_final = bTl_ls[:, -1, :].mean(0)
        eval_info = {"Cost Mean": l_mean, "Cost Final Mean": l_final}

        return self.EvalData(bT_x_plot, bb_Xs, bb_Ys, bb_V, bb_Vdot, bb_u, eval_info)
