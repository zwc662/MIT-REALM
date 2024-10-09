from typing import Callable, NamedTuple

import einops as ei
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from attrs import define
from flax import struct
from loguru import logger

import clfrl.utils.typed_ft as ft
from clfrl.dyn.dyn_types import BBControl, BBHBool, BBHFloat, BState, BTState, State
from clfrl.dyn.task import Task
from clfrl.ncbf.min_norm_cbf import min_norm_cbf
from clfrl.nclf.min_norm_control import bangbang_control, min_norm_control_interp
from clfrl.nclf.sim_nclf import SimNCLF
from clfrl.networks.mlp import MLP
from clfrl.networks.ncbf import MultiValueFn
from clfrl.networks.network_utils import HidSizes, get_act_from_str
from clfrl.networks.optim import get_default_tx
from clfrl.networks.train_state import TrainState
from clfrl.utils.grad_utils import compute_norm
from clfrl.utils.jax_types import BBBool, BBFloat, FloatScalar, MetricsDict
from clfrl.utils.jax_utils import jax_vmap, rep_vmap
from clfrl.utils.loss_utils import weighted_sum_dict
from clfrl.utils.none import get_or
from clfrl.utils.rng import PRNGKey
from clfrl.utils.schedules import Schedule, as_schedule


@define
class AvoidFixedTrainCfg:
    # Batch size to use when updating.
    batch_size: int
    # 0: No discount. Infinity: V = h.
    lam: float | Schedule
    # Use max for the complementarity condition.
    compl_max: bool = False


@define
class AvoidFixedEvalCfg:
    # How long to rollout during eval.
    eval_rollout_T: int


@define
class AvoidFixedCfg:
    act: str
    lr: Schedule

    hids: HidSizes
    train_cfg: AvoidFixedTrainCfg
    eval_cfg: AvoidFixedEvalCfg

    @property
    def alg_name(self) -> str:
        return "AvoidFixed"


class AvoidFixed(struct.PyTreeNode):
    collect_idx: int
    key: PRNGKey
    V: TrainState[FloatScalar]

    nom_pol: Callable = struct.field(pytree_node=False)
    cfg: AvoidFixedCfg = struct.field(pytree_node=False)
    task: Task = struct.field(pytree_node=False)

    lam_sched: optax.Schedule = struct.field(pytree_node=False)

    class EvalData(NamedTuple):
        bT_x_plot: BTState

        bb_Xs: BBFloat
        bb_Ys: BBFloat
        bbh_V: BBHFloat
        bbh_Vdot: BBHFloat
        bbh_Vdot_disc: BBHFloat
        bbh_hmV: BBHFloat
        bbh_compl: BBHFloat
        bbh_eq_h: BBHBool
        bb_u: BBControl
        info: MetricsDict

    @classmethod
    def create(cls, seed: int, task: Task, cfg: AvoidFixedCfg, nom_pol: Callable) -> "AvoidFixed":
        key, key_V = jr.split(jr.PRNGKey(seed), 2)
        pol_obs, V_obs = np.zeros(task.n_polobs), np.zeros(task.n_Vobs)
        act = get_act_from_str(cfg.act)

        # Define V network.
        V_cls = ft.partial(MLP, cfg.hids, act)
        V_def = MultiValueFn(V_cls, task.nh)
        V_tx = get_default_tx(cfg.lr.make())
        V = TrainState.create_from_def(key_V, V_def, (pol_obs,), V_tx)

        lam_sched = as_schedule(cfg.train_cfg.lam)
        logger.info("Discount lam: {}".format(lam_sched))

        return AvoidFixed(jnp.array(0, dtype=jnp.int32), key, V, nom_pol, cfg, task, lam_sched.make())

    @property
    def train_cfg(self):
        return self.cfg.train_cfg

    @property
    def lam(self) -> FloatScalar:
        return self.lam_sched(self.collect_idx)

    @ft.partial(jax.jit, donate_argnums=0)
    def sample_batch(self):
        key = jr.fold_in(self.key, self.collect_idx)
        b_x0 = self.task.sample_train_x0(key, self.train_cfg.batch_size)
        return self.replace(collect_idx=self.collect_idx + 1), b_x0

    def get_V(self, params, state: State):
        V_obs, _ = self.task.get_obs(state)
        Vh = self.V.apply_fn(params, V_obs)
        return Vh.max()

    def get_Vh(self, params, state: State):
        V_obs, _ = self.task.get_obs(state)
        Vh = self.V.apply_fn(params, V_obs)
        return Vh

    @ft.partial(jax.jit, donate_argnums=0)
    def update(self, b_x: BState, loss_weights: MetricsDict):
        def loss_fn(params):
            V_apply = ft.partial(self.get_Vh, params)

            # 3: V(nonzero) > V(0). Penalize V(0) - V(nonzero) > 0.
            bh_V = jax_vmap(V_apply)(b_x)
            bh_h = jax_vmap(self.task.h_components)(b_x)

            # 4: Compute min norm controller.
            bhx_Vx = jax_vmap(jax.jacobian(V_apply))(b_x)
            b_f = jax_vmap(self.task.f)(b_x)
            b_G = jax_vmap(self.task.G)(b_x)
            b_u = jax_vmap(self.nom_pol)(b_x)

            def compute_terms_h(h_V, hx_Vx, h_h, f, G, u):
                def compute_terms(V, x_Vx, h):
                    now = h - V
                    future = jnp.dot(x_Vx, xdot) - self.lam * (V - h)
                    return now, future

                xdot = f + G @ u
                return jax_vmap(compute_terms)(h_V, hx_Vx, h_h)

            bh_now, bh_future = jax_vmap(compute_terms_h)(bh_V, bhx_Vx, bh_h, b_f, b_G, b_u)
            # Enforce h(x) - V(x) <= 0.
            loss_now_pos = jnp.mean(jnn.relu(bh_now) ** 2)
            acc_now = jnp.mean(bh_now <= 0)

            # Enforce Vdot - lambda * (V - h) <=0.
            loss_future_pos = jnp.mean(jnn.relu(bh_now) ** 2)
            acc_future = jnp.mean(bh_now <= 0)

            # The maximum of both terms should be zero.
            if self.train_cfg.compl_max:
                loss_pde = jnp.mean(jnp.maximum(bh_now, bh_future) ** 2)
            else:
                # Represent the complementarity as a multiplication.
                # If it's positive, clip it to force it to be non-positive.
                loss_pde = jnp.mean((jnp.minimum(0.0, bh_now) * jnp.minimum(0.0, bh_future)) ** 2)

            loss_dict = {
                "Loss/Now": loss_now_pos,
                "Loss/Future": loss_future_pos,
                "Loss/PDE": loss_pde,
            }
            info_dict = {"Acc/V_desc": acc_now, "Acc/Nonzero": acc_future}
            loss = weighted_sum_dict(loss_dict, loss_weights)
            return loss, loss_dict | info_dict

        grads, info = jax.grad(loss_fn, has_aux=True)(self.V.params)
        info["V_grad"] = compute_norm(grads)
        info["anneal/lam"] = self.lam
        gamma = jnp.exp(-self.lam * self.task.dt)
        info["anneal/eff_horizon"] = 1 / (1 - gamma)
        V_new = self.V.apply_gradients(grads)
        return self.replace(V=V_new), info

    def get_cbf_control_all(self, alpha: float, state: State):
        u_lb, u_ub = np.array([-1.0]), np.array([+1.0])
        Vh_apply = ft.partial(self.get_Vh, self.V.params)

        h_V = Vh_apply(state)
        h_Vx = jax.jacobian(Vh_apply)(state)
        f = self.task.f(state)
        G = self.task.G(state)
        u_nom = self.nom_pol(state)

        u, r, info = min_norm_cbf(alpha, u_lb, u_ub, h_V, h_Vx, f, G, u_nom)
        return self.task.chk_u(u), (r, info)

    def get_cbf_control(self, alpha: float, state: State):
        return self.get_cbf_control_all(alpha, state)[0]

    def get_cbf_control_sloped(self, alpha_safe: float, alpha_unsafe: float, state: State, V_shift: float = 1e-3):
        return self.get_cbf_control_sloped_all(alpha_safe, alpha_unsafe, state, V_shift)[0]

    def get_cbf_control_sloped_all(self, alpha_safe: float, alpha_unsafe: float, state: State, V_shift: float = 1e-3):
        u_lb, u_ub = np.array([-1.0]), np.array([+1.0])
        Vh_apply = ft.partial(self.get_Vh, self.V.params)

        h_V = Vh_apply(state)
        h_Vx = jax.jacobian(Vh_apply)(state)
        f = self.task.f(state)
        G = self.task.G(state)
        u_nom = self.nom_pol(state)

        h_V = h_V - V_shift

        is_safe = jnp.all(h_V < 0)
        alpha = jnp.where(is_safe, alpha_safe, alpha_unsafe)

        u, r, info = min_norm_cbf(alpha, u_lb, u_ub, h_V, h_Vx, f, G, u_nom)
        return self.task.chk_u(u), (r, info)

    def get_control_interp(self, interp: float, state: State, alpha: float = 0.1):
        """Interpolates between min_norm (0) and bang-bang (1) in Vdot space."""
        u_lb, u_ub = np.array([-1.0]), np.array([+1.0])
        V_apply = ft.partial(self.get_V, self.V.params)

        V = V_apply(state)
        Vx = jax.grad(V_apply)(state)
        f = self.task.f(state)
        G = self.task.G(state)
        # TODO: Make a min norm for CBF.
        u, _, _ = min_norm_control_interp(alpha, interp, u_lb, u_ub, V, Vx, f, G)
        return self.task.chk_u(u)

    def get_control(self, state: State, alpha: float = 0.1):
        return self.get_control_interp(0.0, state, alpha)

    def get_opt_u(self, state: State, alpha: float = 100.0):
        u_lb, u_ub = np.array([-1.0]), np.array([+1.0])
        Vh_apply = ft.partial(self.get_Vh, self.V.params)
        h_V = Vh_apply(state)
        h_Vx = jax.jacobian(Vh_apply)(state)
        f = self.task.f(state)
        G = self.task.G(state)
        Lf_V = ei.einsum(h_Vx, f, "h nx, nx -> h")
        h_LG_V = ei.einsum(h_Vx, G, "h nx, nx nu -> h nu")

        h_Vdot_lb = Lf_V + h_LG_V @ u_lb + alpha * h_V
        Vdot_lb = jnp.max(h_Vdot_lb)

        h_Vdot_ub = Lf_V + h_LG_V @ u_ub + alpha * h_V
        Vdot_ub = jnp.max(h_Vdot_ub)

        # min_u max_i Vdot_i
        return jnp.where(Vdot_lb < Vdot_ub, u_lb, u_ub)

    @ft.partial(jax.jit, static_argnames="T")
    def get_bb_V_nom(self, T: int = None):
        return self.V_for_pol(self.nom_pol, T)

    @ft.partial(jax.jit, static_argnames="T")
    def get_bb_V_opt(self, T: int = None):
        return self.V_for_pol(self.get_opt_u, T)

    def get_bb_V_sloped(
        self, alpha_safe: float = 0.1, alpha_unsafe: float = 100.0, V_shift: float = 1e-3, T: int = None
    ):
        pol = ft.partial(self.get_cbf_control_sloped, alpha_safe, alpha_unsafe, V_shift=V_shift)
        return self.V_for_pol(pol, T)

    def V_for_pol(self, pol, T: int = None):
        T = get_or(T, self.cfg.eval_cfg.eval_rollout_T)
        sim = SimNCLF(self.task, pol, T)
        bb_x, bb_Xs, bb_Ys = self.task.get_contour_x0()
        bbT_x, _, _ = rep_vmap(sim.rollout_plot, rep=2)(bb_x)
        bbT_h = rep_vmap(self.task.h, rep=3)(bbT_x)
        return bbT_h.max(-1)

    @jax.jit
    def eval(self) -> EvalData:
        # Get states for plotting and for metrics.
        b_x0_plot = self.task.get_plot_x0()
        b_x0_metric = self.task.get_metric_x0()
        b_x0_rng = self.task.get_plot_rng_x0()

        # Rollout using the min norm controller.
        alpha_safe, alpha_unsafe = 5.0, 10.0
        nom_pol = ft.partial(self.get_cbf_control_sloped, alpha_safe, alpha_unsafe)
        sim = SimNCLF(self.task, nom_pol, self.cfg.eval_cfg.eval_rollout_T)

        bT_x_plot, _, _ = jax_vmap(sim.rollout_plot)(b_x0_plot)
        bT_x_metric, _ = jax_vmap(sim.rollout)(b_x0_metric)

        def get_V_info(state):
            V_apply = ft.partial(self.get_V, self.V.params)
            Vh_apply = ft.partial(self.get_Vh, self.V.params)
            h_V = Vh_apply(state)
            h_Vx = jax.jacobian(Vh_apply)(state)
            f, G = self.task.f(state), self.task.G(state)

            u_nom = self.nom_pol(state)
            is_safe = jnp.all(h_V < 0)
            alpha = jnp.where(is_safe, alpha_safe, alpha_unsafe)
            u, _, _ = min_norm_cbf(alpha, u_lb, u_ub, h_V, h_Vx, f, G, u_nom)
            xdot = f + jnp.sum(G * u, axis=-1)
            h_Vdot = jnp.sum(h_Vx * xdot, axis=-1)

            h_h = self.task.h_components(state)

            h_Vdot_disc = h_Vdot - self.lam * (h_V - h_h)
            h_hmV = h_h - h_V
            h_compl = jnp.maximum(h_Vdot_disc, h_hmV)
            h_eqh = h_Vdot_disc < h_hmV

            return h_V, h_Vdot, h_Vdot_disc, h_hmV, h_compl, h_eqh, u

        u_lb, u_ub = np.array([-1]), np.array([+1])
        bb_x, bb_Xs, bb_Ys = self.task.get_contour_x0()
        bbh_V, bbh_Vdot, bbh_Vdot_disc, bbh_hmV, bbh_compl, bbh_eq_h, bb_u = rep_vmap(get_V_info, rep=2)(bb_x)

        # bTl_ls = rep_vmap(self.task.l_components, rep=2)(bT_x_metric)
        bT_hs = rep_vmap(self.task.h, rep=2)(bT_x_metric)

        h_mean = bT_hs.mean(-1).mean(0)
        h_max = bT_hs.max(-1).mean(0)
        eval_info = {
            "Constr Mean": h_mean,
            "Constr Max Mean": h_max,
            "Safe Frac": jnp.mean(jnp.all(bT_hs < 0, axis=-1)),
        }

        return self.EvalData(
            bT_x_plot, bb_Xs, bb_Ys, bbh_V, bbh_Vdot, bbh_Vdot_disc, bbh_hmV, bbh_compl, bbh_eq_h, bb_u, eval_info
        )
