from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from attrs import define
from flax import struct
from loguru import logger

import clfrl.utils.typed_ft as ft
from clfrl.dyn.dyn_types import BBControl, BBHFloat, BState, BTState, State
from clfrl.dyn.task import Task
from clfrl.ncbf.min_norm_cbf import min_norm_cbf
from clfrl.nclf.sim_nclf import SimNCLF
from clfrl.networks.mlp import MLP
from clfrl.networks.ncbf import SingleValueFn
from clfrl.networks.network_utils import HidSizes, get_act_from_str
from clfrl.networks.optim import get_default_tx
from clfrl.networks.train_state import TrainState
from clfrl.utils.grad_utils import compute_norm
from clfrl.utils.jax_types import BBFloat, FloatScalar, MetricsDict
from clfrl.utils.jax_utils import jax_vmap, rep_vmap
from clfrl.utils.loss_utils import weighted_sum_dict
from clfrl.utils.none import get_or
from clfrl.utils.rng import PRNGKey
from clfrl.utils.schedules import Schedule


@define
class NCBFTrainCfg:
    # Batch size to use when updating.
    batch_size: int
    # Class kappa constant.
    lam: float

    # x is unsafe   =>   V(x) > epsilon
    unsafe_eps: float
    # x = 0         =>   V(x) < -epsilon
    safe_eps: float

    # Use information about the equilibrium state for safe. Otherwise, use the asserted safety.
    use_eq_state: str = "eq_state"


@define
class NCBFEvalCfg:
    # How long to rollout during eval.
    eval_rollout_T: int


@define
class NCBFCfg:
    act: str
    lr: Schedule

    hids: HidSizes
    train_cfg: NCBFTrainCfg
    eval_cfg: NCBFEvalCfg

    @property
    def alg_name(self) -> str:
        return "NCBF"


class NCBF(struct.PyTreeNode):
    collect_idx: int
    key: PRNGKey
    V: TrainState[FloatScalar]

    nom_pol: Callable = struct.field(pytree_node=False)
    cfg: NCBFCfg = struct.field(pytree_node=False)
    task: Task = struct.field(pytree_node=False)

    class EvalData(NamedTuple):
        bT_x_plot: BTState

        bb_Xs: BBFloat
        bb_Ys: BBFloat
        bb_V: BBHFloat
        bb_Vdot: BBHFloat
        bb_u: BBControl
        info: MetricsDict

    @classmethod
    def create(cls, seed: int, task: Task, cfg: NCBFCfg, nom_pol: Callable) -> "NCBF":
        key, key_V = jr.split(jr.PRNGKey(seed), 2)
        pol_obs, V_obs = np.zeros(task.n_polobs), np.zeros(task.n_Vobs)
        act = get_act_from_str(cfg.act)

        # Define V network.
        V_cls = ft.partial(MLP, cfg.hids, act)
        V_def = SingleValueFn(V_cls)
        V_tx = get_default_tx(cfg.lr.make())
        V = TrainState.create_from_def(key_V, V_def, (pol_obs,), V_tx)

        logger.info("Class kappa lam: {}".format(cfg.train_cfg.lam))

        return NCBF(jnp.array(0, dtype=jnp.int32), key, V, nom_pol, cfg, task)

    @property
    def train_cfg(self):
        return self.cfg.train_cfg

    @property
    def lam(self):
        return self.train_cfg.lam

    @ft.partial(jax.jit, donate_argnums=0)
    def sample_batch(self):
        key = jr.fold_in(self.key, self.collect_idx)
        b_x0 = self.task.sample_train_x0(key, self.train_cfg.batch_size)
        return self.replace(collect_idx=self.collect_idx + 1), b_x0

    def get_V(self, state: State, params=None):
        params = get_or(params, self.V.params)
        V_obs, _ = self.task.get_obs(state)
        return self.V.apply_fn(params, V_obs)

    @ft.partial(jax.jit, donate_argnums=0)
    def update(self, b_x: BState, loss_weights: MetricsDict):
        def loss_fn(params):
            V_apply = ft.partial(self.get_V, params=params)
            b_V = jax_vmap(V_apply)(b_x)

            b_is_unsafe = jax_vmap(self.task.h)(b_x) > 0

            # 1: h(x)  > 0  is unsafe  =>    V(x) > eps <=> eps - V(x) < 0.  Penalize eps - V(x) > 0
            loss_unsafe = jnp.maximum(self.train_cfg.unsafe_eps - b_V, 0.0)
            loss_unsafe = jnp.mean(jnp.where(b_is_unsafe, loss_unsafe, 0.0) ** 2)
            # p( V(x) > 0  |  h(x) > 0 )
            acc_unsafe = jnp.sum((b_V > 0) & b_is_unsafe) / jnp.sum(b_is_unsafe)

            # 2: x = 0      =>     V(x) < 0.0,    penalize V(x) + eps > 0.
            if self.cfg.train_cfg.use_eq_state == "eq_state":
                assert self.task.has_eq_state(), "Should have eq_state!"
                V_0 = V_apply(self.task.eq_state())
                loss_safe = jnp.maximum(0.0, V_0 + self.train_cfg.safe_eps) ** 2
                acc_safe = V_0 < 0.0
            elif self.cfg.train_cfg.use_eq_state == "assert_is_safe":
                b_is_safe = jax_vmap(self.task.assert_is_safe)(b_x)

                loss_safe = jnp.maximum(0.0, b_V + self.train_cfg.safe_eps)
                loss_safe = jnp.mean(jnp.where(b_is_safe, loss_safe, 0.0) ** 2)
                acc_safe = jnp.sum((b_V < 0) & b_is_safe) / jnp.sum(b_is_safe)
            elif self.cfg.train_cfg.use_eq_state == "none":
                loss_safe = 0.0
                acc_safe = 1.0

            # 3: Descent condition. Penalize positive residuals.
            b_Vx = jax_vmap(jax.grad(V_apply))(b_x)
            b_f = jax_vmap(self.task.f)(b_x)
            b_G = jax_vmap(self.task.G)(b_x)
            b_u_nom = jax_vmap(self.nom_pol)(b_x)

            u_lb, u_ub = self.task.u_min, self.task.u_max
            b_u, b_resid, _ = jax_vmap(ft.partial(min_norm_cbf, self.train_cfg.lam, u_lb, u_ub))(
                b_V, b_Vx, b_f, b_G, b_u_nom
            )
            loss_resid = jnp.mean(jnp.maximum(b_resid, 0.0) ** 2)

            b_xdot = b_f + jnp.sum(b_G * b_u[:, None, :], axis=-1)
            b_Vdot = jnp.sum(b_Vx * b_xdot, axis=-1)
            acc_descent = jnp.mean(b_Vdot < 0)

            loss_dict = {
                "Loss/Unsafe": loss_unsafe,
                "Loss/Safe": loss_safe,
                "Loss/Descent": loss_resid,
            }
            info_dict = {"Acc/Unsafe": acc_unsafe, "Acc/Safe": acc_safe, "Acc/Descent": acc_descent}
            loss = weighted_sum_dict(loss_dict, loss_weights)
            return loss, loss_dict | info_dict

        grads, info = jax.grad(loss_fn, has_aux=True)(self.V.params)
        info["V_grad"] = compute_norm(grads)
        V_new = self.V.apply_gradients(grads)
        return self.replace(V=V_new), info

    def get_cbf_control_all(self, alpha: float, state: State):
        V = self.get_V(state)
        Vx = jax.jacobian(self.get_V)(state)
        f = self.task.f(state)
        G = self.task.G(state)
        u_nom = self.nom_pol(state)

        u, r, info = min_norm_cbf(alpha, self.task.u_min, self.task.u_max, V, Vx, f, G, u_nom)
        return self.task.chk_u(u), (r, info)

    def get_cbf_control(self, alpha: float, state: State):
        return self.get_cbf_control_all(alpha, state)[0]

    def get_cbf_control_sloped(self, alpha_safe: float, alpha_unsafe: float, state: State, V_shift: float = 1e-3):
        return self.get_cbf_control_sloped_all(alpha_safe, alpha_unsafe, state, V_shift)[0]

    def get_cbf_control_sloped_all(self, alpha_safe: float, alpha_unsafe: float, state: State, V_shift: float = 1e-3):
        u_lb, u_ub = self.task.u_min, self.task.u_max
        V_apply = ft.partial(self.get_h, self.V.params)

        V = V_apply(state)
        Vx = jax.grad(V_apply)(state)
        f = self.task.f(state)
        G = self.task.G(state)
        u_nom = self.nom_pol(state)

        V = V - V_shift

        is_safe = V < 0
        alpha = jnp.where(is_safe, alpha_safe, alpha_unsafe)

        u, r, info = min_norm_cbf(alpha, u_lb, u_ub, V, Vx, f, G, u_nom)
        return self.task.chk_u(u), (r, info)

    @jax.jit
    def eval(self) -> EvalData:
        # Get states for plotting and for metrics.
        b_x0_plot = self.task.get_plot_x0()
        b_x0_metric = self.task.get_metric_x0()
        b_x0_rng = self.task.get_plot_rng_x0()

        # Rollout using the min norm controller.
        alpha = 5.0
        sim = SimNCLF(self.task, ft.partial(self.get_cbf_control, alpha), self.cfg.eval_cfg.eval_rollout_T)

        bT_x_plot, _, _ = jax_vmap(sim.rollout_plot)(b_x0_plot)
        bT_x_metric, _ = jax_vmap(sim.rollout)(b_x0_metric)

        def get_V_info(state):
            V = self.get_V(state)
            Vx = jax.grad(self.get_V)(state)
            f, G = self.task.f(state), self.task.G(state)

            u_nom = self.nom_pol(state)
            u, _, _ = min_norm_cbf(alpha, u_lb, u_ub, V, Vx, f, G, u_nom)
            xdot = f + jnp.sum(G * u, axis=-1)
            Vdot = jnp.sum(Vx * xdot, axis=-1)

            return V, Vdot, u

        u_lb, u_ub = self.task.u_min, self.task.u_max
        bb_x, bb_Xs, bb_Ys = self.task.get_contour_x0()
        bb_V, bb_Vdot, bb_u = rep_vmap(get_V_info, rep=2)(bb_x)

        bT_hs = rep_vmap(self.task.h, rep=2)(bT_x_metric)

        h_mean = bT_hs.mean(-1).mean(0)
        h_max = bT_hs.max(-1).mean(0)
        eval_info = {
            "Constr Mean": h_mean,
            "Constr Max Mean": h_max,
            "Safe Frac": jnp.mean(jnp.all(bT_hs < 0, axis=-1)),
        }

        return self.EvalData(bT_x_plot, bb_Xs, bb_Ys, bb_V, bb_Vdot, bb_u, eval_info)
