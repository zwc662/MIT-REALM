from typing import Callable, NamedTuple

import einops as ei
import jax
import jax.debug as jd
import jax.lax as lax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import optax
from attrs import define
from flax import struct
from loguru import logger

# from pncbf.ncbf.min_norm_cbf import min_norm_cbf
import pncbf.ncbf.min_norm_cbf as cbf_old
import pncbf.utils.typed_ft as ft
from pncbf.dyn.dyn_types import (
    BBControl,
    BBHBool,
    BBHFloat,
    BControl,
    BHFloat,
    BState,
    BTControl,
    BTHFloat,
    BTState,
    HFloat,
    State,
    Control
)
from pncbf.dyn.sim_cts import SimCtsReal
from pncbf.dyn.task import Task
from pncbf.ncbf.compute_disc_avoid import (
    AllDiscAvoidTerms,
    DiscAvoidTerms,
    compute_all_disc_avoid_terms,
    compute_disc_avoid_terms,
)
from pncbf.networks.block import TmpNet

# from pncbf.nclf.sim_nclf import SimNCLF
from pncbf.networks.ensemble import Ensemble, subsample_ensemble
from pncbf.networks.mlp import MLP
from pncbf.networks.ncbf import MultiNormValueFn, MultiValueFn, Rescale
from pncbf.networks.network_utils import HidSizes, get_act_from_str
from pncbf.networks.optim import get_default_tx
from pncbf.networks.train_state import TrainState
from pncbf.qp.min_norm_cbf import min_norm_cbf, min_norm_cbf_qp_mats
from pncbf.utils.grad_utils import compute_norm, empty_grad_tx, compute_kernel_matrix, compute_kernel_matrix, compute_kernel_gradient, svgd_update  
from pncbf.utils.jax_types import BBBool, BBFloat, BBool, BFloat, BHBool, BTHBool, FloatScalar, MetricsDict
from pncbf.utils.jax_utils import jax_vmap, merge01, rep_vmap, tree_copy, tree_split_dims
from pncbf.utils.loss_utils import weighted_sum_dict
from pncbf.utils.mathtext import from_mathtext
from pncbf.utils.none import get_or
from pncbf.utils.rng import PRNGKey
from pncbf.utils.schedules import Schedule, as_schedule


@define
class IntAvoidTrainCfg:
    # How many trajectories to collect
    collect_size: int
    # What should the spacing be in dt in the trajectory we return?
    # This is different from the dt used in the solver.
    rollout_dt: float
    # How many of the rollout_dt we use.
    rollout_T: int

    # Batch size during training.
    batch_size: int
    # 0: No discount. Infinity: V = h.
    lam: Schedule

    # Target V smoothing. Small values = more smoothing.
    tau: float

    # Instead of discount, we can just "discount" the target. This is biased but so is discount.
    tgt_rhs: Schedule
    # If true, then use Vh(0) = h(0) as a boundary condition.
    use_eq_state: bool = False
    # If true, then record the grad terms. Otherwise, we don't even need to compute grad.
    use_grad_terms: bool = True
    # If V = h, then enforce the gradients also match.
    use_hgrad: bool = False


@define
class IntAvoidEvalCfg:
    # How long to rollout during eval.
    eval_rollout_T: int


@define
class IntAvoidCfg:
    act: str
    lr: Schedule
    wd: Schedule

    hids: HidSizes
    train_cfg: IntAvoidTrainCfg
    eval_cfg: IntAvoidEvalCfg

    # Num in ensemble.
    n_Vs: int
    # Num to use as target.
    n_min_tgt: int

    use_multi_norm: bool = False #False
    rescale_outputs: bool = False

    new_net: bool = False

    @property
    def alg_name(self) -> str:
        return "IntAvoid"


class IntAvoid(struct.PyTreeNode):
    collect_idx: int
    update_idx: int
    key: PRNGKey
    Vh: TrainState[HFloat]
    Vh_tgt: TrainState[HFloat]

    nom_pol: Callable = struct.field(pytree_node=False)
    cfg: IntAvoidCfg = struct.field(pytree_node=False)
    task: Task = struct.field(pytree_node=False)

    _lam: FloatScalar
    lam_sched: optax.Schedule = struct.field(pytree_node=False)
    tgt_rhs_sched: optax.Schedule = struct.field(pytree_node=False)

    class CollectData(NamedTuple):
        """Data collected here has a spacing of train_cfg.rollout_dt."""

        bT_x: BTState
        bT_u: BTControl
        # b_vterms: DiscAvoidTerms
        b_vterms: AllDiscAvoidTerms
        # If true, then V = h. We can enforce derivative conditions on this.
        # This can be modified depending on the value of V(x_T).
        bTh_iseqh: BTHBool

    class Batch(NamedTuple):
        b_x0: BState
        b_u0: BTControl
        b_xT: BState
        bh_iseqh: BBool
        bh_lhs: BHFloat
        bh_int_rhs: BHFloat
        b_discount_rhs: BFloat

    class EvalData(NamedTuple):
        bT_x_plot: BTState

        bb_Xs: BBFloat
        bb_Ys: BBFloat
        bbh_V: BBHFloat
        bbh_Vdot: BBHFloat
        bbh_Vdot_disc: BBHFloat
        # bbh_hmV: BBHFloat
        # bbh_compl: BBHFloat
        # bbh_eq_h: BBHBool
        bb_u: BBControl
        info: MetricsDict

    @classmethod
    def create(cls, seed: int, task: Task, cfg: IntAvoidCfg, nom_pol: Callable) -> "IntAvoid":
        key, key_Vh = jr.split(jr.PRNGKey(seed), 2)
        pol_obs, V_obs = np.zeros(task.n_polobs), np.zeros(task.n_Vobs)
        act = get_act_from_str(cfg.act)

        # Define V network.
        if cfg.new_net:
            Vh_cls = ft.partial(TmpNet, 2, cfg.hids[0], act)
        else:
            Vh_cls = ft.partial(MLP, cfg.hids, act)

        if cfg.use_multi_norm:
            Vh_cls = ft.partial(MultiNormValueFn, Vh_cls, task.nh, 2 * task.nx)
        else:
            Vh_cls = ft.partial(MultiValueFn, Vh_cls, task.nh)

        if cfg.rescale_outputs:
            logger.info("Rescaling outputs! h∈[{}, {}]".format(task.h_min, task.h_max))
            Vh_cls = ft.partial(Rescale, Vh_cls, task.h_min, task.h_max)

        Vh_def = Ensemble(Vh_cls, cfg.n_Vs)
        Vh_tx = get_default_tx(cfg.lr.make(), cfg.wd.make())
        Vh = TrainState.create_from_def(key_Vh, Vh_def, (pol_obs,), Vh_tx)

        # Target network for stability.
        Vh_tgt_def = Ensemble(Vh_cls, num=cfg.n_min_tgt)
        Vh_params_copy = tree_copy(Vh.params)  # Copy so that we can donate buffer.
        Vh_tgt = TrainState.create(apply_fn=Vh_tgt_def.apply, params=Vh_params_copy, tx=empty_grad_tx())

        lam_sched = as_schedule(cfg.train_cfg.lam)
        logger.info("Discount lam: {}".format(lam_sched))

        col_idx, upd_idx = jnp.array(0, dtype=jnp.int32), jnp.array(0, dtype=jnp.int32)
        lam = jnp.array(0.0)
        return IntAvoid(
            col_idx, upd_idx, key, Vh, Vh_tgt, nom_pol, cfg, task, lam, lam_sched.make(), cfg.train_cfg.tgt_rhs.make()
        )

    @property
    def train_cfg(self):
        return self.cfg.train_cfg

    @property
    def lam(self) -> FloatScalar:
        # return self.lam_sched(self.update_idx)
        return self._lam

    @property
    def V_shift(self):
        return -(1 - np.exp(-self.lam * self.task.max_ttc)) * self.task.h_min

    def update_lam(self):
        new_lam = self.lam_sched(self.update_idx)
        return self.replace(_lam=new_lam), new_lam != self._lam

    @property
    def tgt_rhs_coeff(self) -> FloatScalar:
        return self.tgt_rhs_sched(self.update_idx)

    def get_V(self, state: State, params=None):
        return self.get_Vh(state, params).max(-1)

    def get_eh_Vh(self, state: State, params=None):
        params = get_or(params, self.Vh.params)
        V_obs, _ = self.task.get_obs(state)
        eh_Vh = self.Vh.apply_fn(params, V_obs)
        assert eh_Vh.shape == (self.cfg.n_Vs, self.task.nh)
        return eh_Vh

    def get_Vh(self, state: State, params=None):
        eh_Vh = self.get_eh_Vh(state, params)
        h_Vh = eh_Vh.mean(axis=0)
        assert h_Vh.shape == (self.task.nh,)
        return h_Vh

    def get_Vh_tgt(self, state: State):
        return self.get_Vh(state, self.Vh_tgt.params)

    def get_eVh_tgt(self, state: State):
        V_obs, _ = self.task.get_obs(state)
        eh_Vh_tgt = self.Vh.apply_fn(self.Vh_tgt.params, V_obs)
        assert eh_Vh_tgt.shape == (self.cfg.n_Vs, self.task.nh)
        return eh_Vh_tgt

    @ft.partial(jax.jit, donate_argnums=0)
    def sample_dset(self):
        key = jr.fold_in(self.key, self.collect_idx)

        # Sample initial states.
        b_x0 = self.task.sample_train_x0(key, self.train_cfg.collect_size)

        # # Rollout using nominal policy.
        # sim = SimNCLF(self.task, self.nom_pol, self.train_cfg.rollout_T)
        # bT_x, _ = jax_vmap(sim.rollout)(b_x0)

        rollout_dt = self.task.dt
        tf = rollout_dt * (self.train_cfg.rollout_T + 0.001)
        sim = SimCtsReal(self.task, self.nom_pol, tf, rollout_dt, use_obs=False, max_steps=512, use_pid=False)
        bT_x, _, _ = jax_vmap(sim.rollout_plot)(b_x0)
        assert bT_x.shape == (self.train_cfg.collect_size, self.train_cfg.rollout_T + 1, self.task.nx)

        # Compute nominal control (again) here, especially if it's expensive (QP).
        bT_u = rep_vmap(self.nom_pol, rep=2)(bT_x)

        bTh_h = rep_vmap(self.task.h_components, rep=2)(bT_x)

        # Compute value function terms for each trajectory.
        # b_vterms = jax_vmap(ft.partial(compute_disc_avoid_terms, self.lam, self.task.dt))(bTh_h)
        b_vterms: AllDiscAvoidTerms = jax_vmap(ft.partial(compute_all_disc_avoid_terms, self.lam, rollout_dt))(bTh_h)

        if self.train_cfg.use_hgrad:
            bTh_iseqh = b_vterms.Th_max_lhs[:, :-1, :] > b_vterms.Th_max_lhs[:, 1:, :]
            # Set last term to false.
            b1h_iseqh_final = jnp.zeros((self.train_cfg.collect_size, 1, self.task.nh), dtype=bool)
            bTh_iseqh = jnp.concatenate([bTh_iseqh, b1h_iseqh_final], axis=1)
            assert bTh_iseqh.shape == bTh_h.shape
        else:
            bTh_iseqh = None

        return self.replace(collect_idx=self.collect_idx + 1), self.CollectData(bT_x, bT_u, b_vterms, bTh_iseqh)

    @jax.jit
    def get_vterms(self, bT_x: BTState) -> AllDiscAvoidTerms:
        bTh_h = rep_vmap(self.task.h_components, rep=2)(bT_x)
        return jax_vmap(ft.partial(compute_all_disc_avoid_terms, self.lam, self.task.dt))(bTh_h)

    def compute_loss(self, loss_weights, b_x: BState, b_u: BControl, bh_iseqh: BHBool, bh_V_tgt: BHFloat, params):
        batch_size = len(b_x)
        eh_V_apply = ft.partial(self.get_eh_Vh, params=params)

        beh_V_pred = jax_vmap(eh_V_apply)(b_x)
        # behx_Vx = jax_vmap(jax.jacobian(eh_V_apply))(b_x)
        bh_h = jax_vmap(self.task.h_components)(b_x)

        # 1: Value function loss. This applies to all states in the trajectory.
        eh_loss_Vh = jnp.mean((bh_V_tgt[:, None, :] - beh_V_pred) ** 2, axis=0)
        assert eh_loss_Vh.shape == (self.cfg.n_Vs, self.task.nh)
        loss_Vh = jnp.mean(eh_loss_Vh)

        # Penalize underestimating beh_Vpred. i.e., Penalize tgt >= pred.
        eh_loss_Vh_tilt = jnp.mean(jnn.relu(bh_V_tgt[:, None, :] - beh_V_pred) ** 2, axis=0)
        loss_Vh_tilt = jnp.mean(eh_loss_Vh_tilt)

        # 2: Descent loss.
        b_f = jax_vmap(self.task.f)(b_x)
        b_G = jax_vmap(self.task.G)(b_x)

        def compute_terms_h(h_V, x: State, h_h, f, G, u):
            xdot = f + G @ u
            # Compute Vx^T xdot
            _, eh_Vdot = jax.jvp(eh_V_apply, (x,), (xdot,))
            assert eh_Vdot.shape == (self.cfg.n_Vs, self.task.nh)
            future = eh_Vdot - self.lam * (h_V - h_h)
            return future

        grad_terms = {}
        info_dict = {}
        if self.train_cfg.use_grad_terms:
            beh_now = bh_h[:, None, :] - beh_V_pred
            beh_future = jax_vmap(compute_terms_h)(beh_V_pred, b_x, bh_h, b_f, b_G, b_u)
            assert beh_now.shape == beh_future.shape == (batch_size, self.cfg.n_Vs, self.task.nh)
            # Enforce h(x) - V(x) <= 0.
            loss_now_pos = jnp.mean(jnn.relu(beh_now) ** 2)
            acc_now = jnp.mean(beh_now <= 0)

            # Enforce Vdot - lambda * (V - h) <=0.
            loss_future_pos = jnp.mean(jnn.relu(beh_future) ** 2)
            acc_future = jnp.mean(beh_future <= 0)

            # Represent the complementarity as a multiplication.
            # If it's positive, clip it to force it to be non-positive.
            loss_pde = jnp.mean((beh_now * beh_future) ** 2)

            grad_terms = {
                "Loss/Now": loss_now_pos,
                "Loss/Future": loss_future_pos,
                "Loss/PDE": loss_pde,
            }
            info_dict = {"Acc/V_desc": acc_now, "Acc/Nonzero": acc_future}
        else:
            if ("Loss/Future" in loss_weights) or ("Loss/Now" in loss_weights):
                raise ValueError("Set use_grad_terms to True to use Loss/Future or Loss/Now!")

        # 3: Gradient loss for when V = h.
        if self.train_cfg.use_hgrad:
            bh_V_pred = lax.stop_gradient(jnp.mean(beh_V_pred, axis=1))
            bh_iseqh = bh_iseqh & (bh_V_pred <= bh_V_tgt)

            b_xdot = b_f + jnp.sum(b_G * b_u[:, None, :], axis=-1)

            # bh_hdot = jnp.sum(bhx_hgrad * b_xdot, axis=-1)
            _, bh_hdot = jax.vmap(lambda x, xdot: jax.jvp(self.task.h_components, (x,), (xdot,)))(b_x, b_xdot)
            assert bh_hdot.shape == (batch_size, self.task.nh)

            _, beh_Vdot = jax.vmap(lambda x, xdot: jax.jvp(eh_V_apply, (x,), (xdot,)))(b_x, b_xdot)
            assert beh_Vdot.shape == (batch_size, self.cfg.n_Vs, self.task.nh)

            bh_loss_dV_eqh = jnp.mean((beh_Vdot - bh_hdot[:, None, :]) ** 2, axis=1)
            assert bh_loss_dV_eqh.shape == (batch_size, self.task.nh)

            bh_loss_dV_eq0 = jnp.mean(beh_Vdot**2, axis=1)
            assert bh_loss_dV_eq0.shape == (batch_size, self.task.nh)

            bh_loss_dV = jnp.where(bh_iseqh, bh_loss_dV_eqh, bh_loss_dV_eq0)
            loss_dV = jnp.mean(bh_loss_dV)

            grad_terms = {"Loss/dV_mse": loss_dV}

        extra_loss = {}
        if self.train_cfg.use_eq_state:
            assert self.task.has_eq_state()
            x_eq = self.task.eq_state()
            eh_V_pred_eq = eh_V_apply(x_eq)
            h_eq = self.task.h_components(x_eq)
            loss_Vh_eq = jnp.mean((h_eq - eh_V_pred_eq) ** 2)
            extra_loss["Loss/Equil"] = loss_Vh_eq

        loss_dict = {
            "Loss/Vh_mse": loss_Vh,
            "Loss/Vh_tilt": loss_Vh_tilt,
            **grad_terms,
            # "Loss/Hess Fro": loss_fro,
            **extra_loss,
        }
        loss = weighted_sum_dict(loss_dict, loss_weights)
        return loss, loss_dict | info_dict

    def update_Vh(self, loss_weights, b_x: BState, b_u: BControl, bh_iseqh: BHBool, bh_V_tgt: BHFloat):
        loss_fn = ft.partial(self.compute_loss, loss_weights, b_x, b_u, bh_iseqh, bh_V_tgt)
        grads, info = jax.grad(loss_fn, has_aux=True)(self.Vh.params)
        info["V_grad"] = compute_norm(grads)
        Vh_new = self.Vh.apply_gradients(grads)

        return self.replace(Vh=Vh_new), info

    @ft.partial(jax.jit, donate_argnums=0)
    def update(self, data: Batch, loss_weights: MetricsDict):
        b_size = len(data.b_xT)
        b_Vobs_T, _ = jax_vmap(self.task.get_obs)(data.b_xT)

        # 1: Compute value function at all states for each trajectory.
        # 1.1: Eval V tgt at xT.
        # bh_VT = self.get_Vh(b_xT)
        key_update = jr.fold_in(self.key, self.update_idx)
        key_tgt = key_update
        target_params = subsample_ensemble(key_tgt, self.Vh_tgt.params, self.cfg.n_min_tgt, self.cfg.n_min_tgt)
        ebh_VhT = self.Vh_tgt.apply_fn(target_params, b_Vobs_T)
        # bh_VhT = jnp.mean(ebh_VhT, axis=0)
        bh_VhT = jnp.min(ebh_VhT, axis=0)
        # bh_VhT = jnp.full((b_size, self.task.nh), -1e12)

        # Speed up tgt 1: clip it between hmin and hmax.
        bh_VhT = jnp.clip(bh_VhT, self.task.h_min, self.task.h_max)
        # Speed up tgt 2: If its smaller than h, then make it h.
        bh_tgt = jax_vmap(self.task.h_components)(data.b_xT)
        bh_VhT = jnp.maximum(bh_VhT, bh_tgt)

        # 1.2: Compute (target) V at T_x using b_vterms
        bh_rhs = data.bh_int_rhs + data.b_discount_rhs[:, None] * bh_VhT
        bh_V_tgt = data.bh_lhs + self.tgt_rhs_coeff * jnp.maximum(0, bh_rhs - data.bh_lhs)
        assert bh_V_tgt.shape == (b_size, self.task.nh)

        new_self, info_mean = self.update_Vh(loss_weights, data.b_x0, data.b_u0, data.bh_iseqh, bh_V_tgt)

        info_mean["anneal/lam"] = self.lam
        gamma = jnp.exp(-self.lam * self.task.dt)
        info_mean["anneal/eff_horizon"] = 1 / (1 - gamma)
        info_mean["anneal/tgt_rhs_coeff"] = self.tgt_rhs_coeff
        info_mean["collect_idx"] = self.collect_idx
        info_mean["update_idx"] = self.update_idx

        Vh_tgt_params = optax.incremental_update(new_self.Vh.params, self.Vh_tgt.params, self.train_cfg.tau)
        Vh_tgt = self.Vh_tgt.replace(params=Vh_tgt_params)

        return new_self.replace(Vh_tgt=Vh_tgt, update_idx=self.update_idx + 1), info_mean

    def get_cbf_qpmats(self, alpha_safe: float, alpha_unsafe: float, state: State, V_shift: float = 1e-3, nom_pol=None):
        u_lb, u_ub = self.task.u_min, self.task.u_max
        Vh_apply = ft.partial(self.get_Vh, params=self.Vh.params)
        nom_pol = get_or(nom_pol, self.nom_pol)

        h_V = Vh_apply(state)
        hx_Vx = jax.jacobian(Vh_apply)(state)
        f = self.task.f(state)
        G = self.task.G(state)
        u_nom = nom_pol(state)

        # Give a small margin.
        h_V = h_V + V_shift

        if isinstance(alpha_safe, float) or alpha_safe.ndim == 0:
            is_safe = jnp.all(h_V < 0)
            alpha = jnp.where(is_safe, alpha_safe, alpha_unsafe)
        else:
            assert alpha_safe.shape == alpha_unsafe.shape == (self.task.nh,)
            h_is_safe = h_V < 0
            h_alpha_safe, h_alpha_unsafe = alpha_safe, alpha_unsafe
            h_alpha = jnp.where(h_is_safe, h_alpha_safe, h_alpha_unsafe)
            alpha = h_alpha

        penalty, relax_eps1, relax_eps2 = 10.0, 5e-1, 20.0
        qp = min_norm_cbf_qp_mats(alpha, u_lb, u_ub, h_V, hx_Vx, f, G, u_nom, penalty, relax_eps1, relax_eps2)
        return qp

    def get_cbf_control_sloped_all(
        self, alpha_safe: float, alpha_unsafe: float, state: State, V_shift: float = 1e-3, nom_pol=None
    ):
        u_lb, u_ub = self.task.u_min, self.task.u_max
        Vh_apply = ft.partial(self.get_Vh, params=self.Vh.params)

        nom_pol = get_or(nom_pol, self.nom_pol)

        h_V = Vh_apply(state)
        h_Vx = jax.jacobian(Vh_apply)(state)
        f = self.task.f(state)
        G = self.task.G(state)
        u_nom = nom_pol(state)

        # Give a small margin.
        h_V = h_V + V_shift

        if isinstance(alpha_safe, float) or alpha_safe.ndim == 0:
            is_safe = jnp.all(h_V < 0)
            alpha = jnp.where(is_safe, alpha_safe, alpha_unsafe)
        else:
            assert alpha_safe.shape == alpha_unsafe.shape == (self.task.nh,)
            h_is_safe = h_V < 0
            h_alpha_safe, h_alpha_unsafe = alpha_safe, alpha_unsafe
            h_alpha = jnp.where(h_is_safe, h_alpha_safe, h_alpha_unsafe)
            alpha = h_alpha

        # u, r, (qp_state, qp_mats) = cbf_old.min_norm_cbf(alpha, u_lb, u_ub, h_V, h_Vx, f, G, u_nom)
        u, r, sol = min_norm_cbf(alpha, u_lb, u_ub, h_V, h_Vx, f, G, u_nom)
        return self.task.chk_u(u), (r, sol)

    def get_cbf_control_sloped(
        self, alpha_safe: float, alpha_unsafe: float, state: State, V_shift: float = 1e-3, nom_pol=None
    ):
        return self.get_cbf_control_sloped_all(alpha_safe, alpha_unsafe, state, V_shift, nom_pol)[0]

    def get_cbf_control(self, alpha: float, state: State):
        return self.get_cbf_control_sloped_all(alpha, alpha, state)[0]

    @ft.partial(jax.jit, static_argnames=["T", "setup_idx", "use_pid"])
    def get_bb_V_nom(self, T: int = None, setup_idx: int = 0, use_pid: bool = False):
        return self.V_for_pol(self.nom_pol, T, setup_idx, use_pid)

    # @ft.partial(jax.jit, static_argnames=["T", "setup_idx"])
    # def get_bb_V_opt(self, T: int = None, setup_idx: int = 0):
    #     return self.V_for_pol(self.get_opt_u, T, setup_idx)

    @ft.partial(jax.jit, static_argnames=["T", "setup_idx"])
    def get_bb_V_sloped(self, T: int = None, setup_idx: int = 0):
        nom_pol = ft.partial(self.get_cbf_control_sloped, 2.0, 100.0)
        return self.V_for_pol(nom_pol, T, setup_idx, use_pid=False)

    def V_for_pol(self, pol, T: int = None, setup_idx: int = 0, use_pid: bool = True):
        T = get_or(T, self.cfg.eval_cfg.eval_rollout_T)
        tf = self.task.dt * (T + 0.001)
        sim = SimCtsReal(self.task, pol, tf, self.task.dt, use_obs=False, use_pid=use_pid, max_steps=T + 3)
        bb_x, bb_Xs, bb_Ys = self.task.get_contour_x0(setup_idx)
        bbT_x, _, _ = rep_vmap(sim.rollout_plot, rep=2)(bb_x)
        bbT_h = rep_vmap(self.task.h, rep=3)(bbT_x)
        return bbT_h.max(-1)

    @jax.jit
    def eval(self) -> EvalData:
        # Get states for plotting and for metrics.
        b_x0_plot = self.task.get_plot_x0(0)
        b_x0_metric = self.task.get_metric_x0()
        # b_x0_loss = self.task.get_loss_x0()

        # Rollout using the min norm controller.
        alpha_safe, alpha_unsafe = 2.0, 100.0
        pol = ft.partial(self.get_cbf_control_sloped, alpha_safe, alpha_unsafe)
        # sim = SimNCLF(self.task, pol, self.cfg.eval_cfg.eval_rollout_T)
        eval_rollout_T = self.cfg.eval_cfg.eval_rollout_T
        tf = self.task.dt * eval_rollout_T

        # sim = SimCtsReal(self.task, pol, tf, self.task.dt, use_obs=False)
        # bT_x_plot, _, _ = jax_vmap(sim.rollout_plot)(b_x0_plot)
        # bT_x_metric, _, _ = jax_vmap(sim.rollout_plot)(b_x0_metric)

        # Don't use PID stepsize controller, since the QP controller is probably super nonsmooth.
        sim = SimCtsReal(self.task, pol, tf, self.task.dt, use_obs=False, use_pid=False, max_steps=eval_rollout_T + 1)
        bT_x_plot, _, _ = jax_vmap(sim.rollout_plot)(b_x0_plot)
        bT_x_metric, _, _ = jax_vmap(sim.rollout_plot)(b_x0_metric)

        def get_V_info(state):
            Vh_apply = ft.partial(self.get_Vh, params=self.Vh.params)
            h_V = Vh_apply(state)
            h_Vx = jax.jacobian(Vh_apply)(state)
            f, G = self.task.f(state), self.task.G(state)
            h_h = self.task.h_components(state)

            u_nom = self.nom_pol(state)
            is_safe = jnp.all(h_V < 0)
            alpha = jnp.where(is_safe, alpha_safe, alpha_unsafe)
            # u, _, _ = cbf_old.min_norm_cbf(alpha, u_lb, u_ub, h_V, h_Vx, f, G, u_nom)
            u, _, _ = min_norm_cbf(alpha, u_lb, u_ub, h_V, h_Vx, f, G, u_nom)
            xdot = f + jnp.sum(G * u, axis=-1)
            h_Vdot = jnp.sum(h_Vx * xdot, axis=-1)

            h_Vdot_disc = h_Vdot - self.lam * (h_V - h_h)

            return h_V, h_Vdot, h_Vdot_disc, u

        # def get_loss_info(state):
        #     Vh_apply = ft.partial(self.get_Vh, params=self.Vh.params)
        #     h_V = Vh_apply(state)
        #     h_Vx = jax.jacobian(Vh_apply)(state)
        #     h_h = self.task.h_components(state)
        #
        #     # Check descent condition along pi.
        #     f, G = self.task.f(state), self.task.G(state)
        #     u_nom = self.nom_pol(state)
        #     # is_safe = jnp.all(h_V < 0)
        #     # alpha = jnp.where(is_safe, alpha_safe, alpha_unsafe)
        #     # u_qp, _, _ = cbf_old.min_norm_cbf(alpha, u_lb, u_ub, h_V, h_Vx, f, G, u_nom)
        #     # u_qp, _, _ = min_norm_cbf(alpha, u_lb, u_ub, h_V, h_Vx, f, G, u_nom)
        #
        #     xdot_nom = f + jnp.sum(G * u_nom, axis=-1)
        #     # xdot_qp = f + jnp.sum(G * u_qp, axis=-1)
        #
        #     h_Vdot_nom = jnp.sum(h_Vx * xdot_nom, axis=-1)
        #     # h_Vdot_qp = jnp.sum(h_Vx * xdot_qp, axis=-1)
        #
        #     h_Vdot_disc_nom = h_Vdot_nom - self.lam * (h_V - h_h)
        #
        #     # h_h > 0  =>  h_V > 0
        #     # equiv as  ~( h_h > 0 ) or (h_V > 0)
        #     h_clsfy = (h_h <= 0) | (h_V > 0)
        #
        #     return h_clsfy, h_V - h_h, h_Vdot_nom, h_Vdot_disc_nom

        u_lb, u_ub = self.task.u_min, self.task.u_max
        bb_x, bb_Xs, bb_Ys = self.task.get_contour_x0()
        bbh_V, bbh_Vdot, bbh_Vdot_disc, bb_u = rep_vmap(get_V_info, rep=2)(bb_x)

        # bh_clsfy, bh_hmV, bh_Vdot_nom, bh_Vdot_disc_nom = jax_vmap(get_loss_info)(b_x0_loss)
        # h_desc_nom = jnp.mean(bh_Vdot_nom <= 0, axis=0)
        # h_desc_disc_nom = jnp.mean(bh_Vdot_disc_nom <= 0, axis=0)
        # # h_desc_qp = jnp.mean(bh_Vdot_qp <= 0, axis=0)
        # desc_nom_max = (bh_Vdot_nom.max(axis=1) <= 0).mean()
        # desc_nom_disc_max = (bh_Vdot_disc_nom.max(axis=1) <= 0).mean()
        # # desc_qp_max = (bh_Vdot_qp.max(axis=1) <= 0).mean()
        # h_hmV_neg = jnp.mean(bh_hmV <= 0, axis=0)
        # hmV_neg_max = jnp.mean(bh_hmV.max(axis=1) <= 0, axis=0)
        #
        # h_clsfy = bh_clsfy.mean(axis=0)
        # h_clsfy_max = jnp.all(bh_clsfy, axis=1).mean()

        loss_info = {}
        # for ii, h_label in enumerate(self.task.h_labels_clean):
        #     loss_info[f"Descent/Nom/{h_label}"] = h_desc_nom[ii]
        #     loss_info[f"Descent/Nom Disc/{h_label}"] = h_desc_disc_nom[ii]
        #     # loss_info[f"Descent/QP/{h_label}"] = h_desc_qp[ii]
        #     loss_info[f"ClassifyDet/{h_label}"] = h_hmV_neg[ii]
        #     loss_info[f"Classify/{h_label}"] = h_clsfy[ii]
        # loss_info[f"Descent/Nom/Max"] = desc_nom_max
        # loss_info[f"Descent/Nom Disc/Max"] = desc_nom_disc_max
        # loss_info[f"Descent/QP/Max"] = desc_qp_max
        # loss_info[f"ClassifyDet/Max"] = hmV_neg_max
        # loss_info[f"Classify/Max"] = h_clsfy_max

        # bTl_ls = rep_vmap(self.task.l_components, rep=2)(bT_x_metric)
        bTh_hs = rep_vmap(self.task.h_components, rep=2)(bT_x_metric)
        bT_hs = bTh_hs.max(axis=2)

        h_mean = bT_hs.mean(-1).mean(0)
        h_max = bT_hs.max(-1).mean(0)

        h_labels = self.task.h_labels_clean
        safe_fracs = {
            "Safe/{}".format(h_labels[ii]): jnp.all(bTh_hs[:, :, ii] < 0, axis=1).mean() for ii in range(self.task.nh)
        }

        eval_info = {
            "Constr Mean": h_mean,
            "Constr Max Mean": h_max,
            "Safe Frac": jnp.mean(jnp.all(bT_hs < 0, axis=-1)),
            **safe_fracs,
            **loss_info,
        }

        return self.EvalData(bT_x_plot, bb_Xs, bb_Ys, bbh_V, bbh_Vdot, bbh_Vdot_disc, bb_u, eval_info)



@define
class ModelBasedIntAvoidCfg(IntAvoidCfg):
    n_fs: int = 2
    n_Gs: int = 2
 
class ModelBasedIntAvoid(IntAvoid):
    f: TrainState[HFloat]
    G: TrainState[HFloat]


    class Normalizer(NamedTuple):
        x_mean: State
        x_std: State

        u_mean: Control
        u_std: Control

        def transform_state(self, x: State) -> State:
            return (x - self.x_mean) / self.x_std
        
        def transform_control(self, u: Control) -> Control:
            return (u - self.u_mean) / self.u_std
        
        def recover_state(self, x: State) -> State:
            """Recover the original state from the normalized state."""
            return x * self.x_std + self.x_mean
        
        def recover_control(self, u: Control) -> Control:
            """Recover the original control from the normalized control."""
            return u * self.u_std + self.u_mean
  
        @classmethod
        def create(cls, dset: IntAvoid.CollectData):
            """
            Compute mean and standard deviation for both state and control matrices.
            
            Args:
                states (np.ndarray): B x T x state_dim matrix.
                controls (np.ndarray): B x T x control_dim matrix.
            
            Returns:
                Normalizer: A Normalizer instance with computed means and stds.
            """
            
            x_mean = dset.bT_x.mean(axis=(0, 1))  # Average over batch and time dimensions
            x_std = dset.bT_x.std(axis=(0, 1)) + 1e-8  # To avoid division by zero

            u_mean = dset.bT_u.mean(axis=(0, 1))  # Average over batch and time dimensions
            u_std = dset.bT_u.std(axis=(0, 1)) + 1e-8  # To avoid division by zero

            return cls(x_mean=x_mean, x_std=x_std, u_mean=u_mean, u_std=u_std)
            

    class Batch(NamedTuple):
        b_x0: BState
        b_u0: BTControl
        b_nxt_x0: BState
        b_xT: BState
        bh_iseqh: BBool
        bh_lhs: BHFloat
        bh_int_rhs: BHFloat
        b_discount_rhs: BFloat

    @classmethod
    def create_from_base(cls, baseline: IntAvoid) -> "ModelBasedIntAvoid":
        base_kwargs = {k: getattr(baseline, k) for k in baseline.__dataclass_fields__}
        
        state = np.zeros(baseline.task.nx)
        act = get_act_from_str(baseline.cfg.act)
        

        base_kwargs = {k: getattr(baseline, k) for k in baseline.__dataclass_fields__}

        state = np.zeros(baseline.task.nx)
        act = get_act_from_str(baseline.cfg.act)
 
        key, key_f, key_G = jr.split(base_kwargs.pop('key'), 3)
         
        # Define f and G networks.
        f_cls = ft.partial(MLP, baseline.cfg.hids, act)
        G_cls = ft.partial(MLP, baseline.cfg.hids, act)
 
        f_cls = ft.partial(MultiValueFn, f_cls, baseline.task.nx)
        G_cls = ft.partial(MultiValueFn, G_cls, baseline.task.nx * baseline.task.nu)


        f_def = Ensemble(f_cls, baseline.cfg.n_fs)
        f_tx = get_default_tx(baseline.cfg.lr.make(), baseline.cfg.wd.make())
        f = TrainState.create_from_def(key_f, f_def, (state,), f_tx)
 
        G_def = Ensemble(G_cls, baseline.cfg.n_Gs)
        G_tx = get_default_tx(baseline.cfg.lr.make(), baseline.cfg.wd.make())
        G = TrainState.create_from_def(key_G, G_def, (state,), G_tx)

        return ModelBasedIntAvoid(key = key, f = f, G = G, **base_kwargs)
 


    @classmethod
    def create(cls, seed: int, task: Task, cfg: ModelBasedIntAvoidCfg, nom_pol: Callable) -> "ModelBasedIntAvoid":
        baseline = super(ModelBasedIntAvoid, cls).create(seed, task, cfg, nom_pol)
        return cls.create_from_base(baseline)
        
    def get_f(self, state: State, params=None): #,normalizer: Normalizer):
        params = get_or(params, self.f.params)
        #normalized_state = normalizer.transform_state(state) 
        #assert normalized_state.shape == (self.task.nx,)
        f = self.f.apply_fn(params, state) #normalized_state(state))
        assert f.shape == (self.cfg.n_fs, self.task.nx)
        return f.mean(axis = 0)
 
    def get_G(self, state: State, params=None): #,normalizer: Normalizer):
        params = get_or(params, self.G.params)
        #normalized_state = normalizer.transform_state(state)  

        #assert normalized_state.shape == (self.task.nx, ) 

        G = self.G.apply_fn(params, state) #normalized_state(state))
        assert G.shape == (self.cfg.n_Gs, self.task.nx * self.task.nu)
        return G.mean(axis = 0).reshape(self.task.nx, self.task.nu)
    
    

    def compute_loss(self, loss_weights, b_x: BState, b_u: BControl, bh_iseqh: BHBool, bh_V_tgt: BHFloat, params):
        batch_size = len(b_x)
        eh_V_apply = ft.partial(self.get_eh_Vh, params=params)

        beh_V_pred = jax_vmap(eh_V_apply)(b_x)
        # behx_Vx = jax_vmap(jax.jacobian(eh_V_apply))(b_x)
        bh_h = jax_vmap(self.task.h_components)(b_x)

        # 1: Value function loss. This applies to all states in the trajectory.
        eh_loss_Vh = jnp.mean((bh_V_tgt[:, None, :] - beh_V_pred) ** 2, axis=0)
        assert eh_loss_Vh.shape == (self.cfg.n_Vs, self.task.nh)
        loss_Vh = jnp.mean(eh_loss_Vh)

        # Penalize underestimating beh_Vpred. i.e., Penalize tgt >= pred.
        eh_loss_Vh_tilt = jnp.mean(jnn.relu(bh_V_tgt[:, None, :] - beh_V_pred) ** 2, axis=0)
        loss_Vh_tilt = jnp.mean(eh_loss_Vh_tilt)


        # 2: Descent loss.
        b_f = jax_vmap(self.get_f)(b_x)
        b_G = jax_vmap(self.get_G)(b_x)

        def compute_terms_h(h_V, x: State, h_h, f, G, u):
            xdot = f + G @ u
            # Compute Vx^T xdot
            _, eh_Vdot = jax.jvp(eh_V_apply, (x,), (xdot,))
            assert eh_Vdot.shape == (self.cfg.n_Vs, self.task.nh)
            future = eh_Vdot - self.lam * (h_V - h_h)
            return future

        grad_terms = {}
        info_dict = {}

        
        if self.train_cfg.use_grad_terms:
            beh_now = bh_h[:, None, :] - beh_V_pred
            beh_future = jax_vmap(compute_terms_h)(beh_V_pred, b_x, bh_h, b_f, b_G, b_u)
            assert beh_now.shape == beh_future.shape == (batch_size, self.cfg.n_Vs, self.task.nh)
            # Enforce h(x) - V(x) <= 0.
            loss_now_pos = jnp.mean(jnn.relu(beh_now) ** 2)
            acc_now = jnp.mean(beh_now <= 0)

            # Enforce Vdot - lambda * (V - h) <=0.
            loss_future_pos = jnp.mean(jnn.relu(beh_future) ** 2)
            acc_future = jnp.mean(beh_future <= 0)

            # Represent the complementarity as a multiplication.
            # If it's positive, clip it to force it to be non-positive.
            loss_pde = jnp.mean((beh_now * beh_future) ** 2)

            grad_terms = {
                "Loss/Now": loss_now_pos,
                "Loss/Future": loss_future_pos,
                "Loss/PDE": loss_pde,
            }
            info_dict = {"Acc/V_desc": acc_now, "Acc/Nonzero": acc_future}
        else:
            if ("Loss/Future" in loss_weights) or ("Loss/Now" in loss_weights):
                raise ValueError("Set use_grad_terms to True to use Loss/Future or Loss/Now!")

        # 3: Gradient loss for when V = h.
        if self.train_cfg.use_hgrad:
            bh_V_pred = lax.stop_gradient(jnp.mean(beh_V_pred, axis=1))
            bh_iseqh = bh_iseqh & (bh_V_pred <= bh_V_tgt)

            b_xdot = b_f + jnp.sum(b_G * b_u[:, None, :], axis=-1)

            # bh_hdot = jnp.sum(bhx_hgrad * b_xdot, axis=-1)
            _, bh_hdot = jax.vmap(lambda x, xdot: jax.jvp(self.task.h_components, (x,), (xdot,)))(b_x, b_xdot)
            assert bh_hdot.shape == (batch_size, self.task.nh)

            _, beh_Vdot = jax.vmap(lambda x, xdot: jax.jvp(eh_V_apply, (x,), (xdot,)))(b_x, b_xdot)
            assert beh_Vdot.shape == (batch_size, self.cfg.n_Vs, self.task.nh)

            bh_loss_dV_eqh = jnp.mean((beh_Vdot - bh_hdot[:, None, :]) ** 2, axis=1)
            assert bh_loss_dV_eqh.shape == (batch_size, self.task.nh)

            bh_loss_dV_eq0 = jnp.mean(beh_Vdot**2, axis=1)
            assert bh_loss_dV_eq0.shape == (batch_size, self.task.nh)

            bh_loss_dV = jnp.where(bh_iseqh, bh_loss_dV_eqh, bh_loss_dV_eq0)
            loss_dV = jnp.mean(bh_loss_dV)

            grad_terms = {"Loss/dV_mse": loss_dV}

        extra_loss = {}
        if self.train_cfg.use_eq_state:
            assert self.task.has_eq_state()
            x_eq = self.task.eq_state()
            eh_V_pred_eq = eh_V_apply(x_eq)
            h_eq = self.task.h_components(x_eq)
            loss_Vh_eq = jnp.mean((h_eq - eh_V_pred_eq) ** 2)
            extra_loss["Loss/Equil"] = loss_Vh_eq

        loss_dict = {
            "Loss/Vh_mse": loss_Vh,
            "Loss/Vh_tilt": loss_Vh_tilt,
            **grad_terms,
            # "Loss/Hess Fro": loss_fro,
            **extra_loss,
        }
        loss = weighted_sum_dict(loss_dict, loss_weights)
        return loss, loss_dict | info_dict

    
    def compute_gap(self, f_params, G_params, b_x: BState, b_u: BControl, b_nxt_x: BState):
        b_f = jax_vmap(ft.partial(self.get_f, params = f_params))(b_x)
        b_G = jax_vmap(ft.partial(self.get_G, params = G_params))(b_x)
 
        # Model loss
        b_dx_pred = jax_vmap(lambda f, G, u: f + G @ u)(b_f, b_G, b_u)
        b_nxt_x_pred = b_dx_pred * self.task.dt + b_x
        loss_b = jnp.mean((b_nxt_x_pred - b_nxt_x) ** 2, axis=0)
        loss = jnp.mean(loss_b)
        return loss, {'Loss/model_mse': loss}


    def update_Vh(self, loss_weights, b_x: BState, b_u: BControl, bh_iseqh: BHBool, bh_V_tgt: BHFloat):
        loss_fn = ft.partial(self.compute_loss, loss_weights, b_x, b_u, bh_iseqh, bh_V_tgt)
        grads, info = jax.grad(loss_fn, has_aux=True)(self.Vh.params)
        info["V_grad"] = compute_norm(grads)
        Vh_new = self.Vh.apply_gradients(grads)
        return Vh_new, info
    
    def update_f_G(self, b_x: BState, b_u: BControl, b_nxt_x: BState):
        gap_fn = ft.partial(self.compute_gap, b_x = b_x, b_u = b_u, b_nxt_x = b_nxt_x)
        (f_grads, G_grads), info = jax.grad(gap_fn, has_aux=True, argnums=(0, 1))(self.f.params, self.G.params)
        
        # Compute the kernel matrix and kernel gradients 
        if self.cfg.n_fs > 2:
            f_ensemble_params = self.f.params['params']['VmapMultiValueFn_0'] 
            f_ensemble_grads = f_grads['params']['VmapMultiValueFn_0']
            f_kernel_matrix = compute_kernel_matrix(f_ensemble_params)
            f_kernel_gradients = compute_kernel_gradient(f_ensemble_params)
            # Compute SVGD updates
            f_ensemble_grads = svgd_update(f_ensemble_grads, f_kernel_matrix, f_kernel_gradients)
            f_grads['params']['VmapMultiValueFn_0'] = f_ensemble_grads 
             
        if self.cfg.n_Gs > 2:
            G_ensemble_params = self.G.params['params']['VmapMultiValueFn_0'] 
            G_ensemble_grads = G_grads['params']['VmapMultiValueFn_0']
            G_kernel_matrix = compute_kernel_matrix(G_ensemble_params)
            G_kernel_gradients = compute_kernel_gradient(G_ensemble_params)
            # Compute SVGD updates
            G_ensemble_grads = svgd_update(G_ensemble_grads, G_kernel_matrix, G_kernel_gradients)
            G_grads['params']['VmapMultiValueFn_0'] = G_ensemble_grads 
            
        info['f_grad'] = compute_norm(f_grads)
        info['G_grad'] = compute_norm(G_grads)
        f_new = self.f.apply_gradients(f_grads)
        G_new = self.G.apply_gradients(G_grads)

        return (f_new, G_new), info

    @ft.partial(jax.jit, donate_argnums=0)
    def update(self, data: Batch, loss_weights: MetricsDict):
        b_size = len(data.b_xT)
        b_Vobs_T, _ = jax_vmap(self.task.get_obs)(data.b_xT)

        # 1: Compute value function at all states for each trajectory.
        # 1.1: Eval V tgt at xT.
        # bh_VT = self.get_Vh(b_xT)
        key_update = jr.fold_in(self.key, self.update_idx)
        key_tgt = key_update
        target_params = subsample_ensemble(key_tgt, self.Vh_tgt.params, self.cfg.n_min_tgt, self.cfg.n_min_tgt)
        ebh_VhT = self.Vh_tgt.apply_fn(target_params, b_Vobs_T)
        # bh_VhT = jnp.mean(ebh_VhT, axis=0)
        bh_VhT = jnp.min(ebh_VhT, axis=0)
        # bh_VhT = jnp.full((b_size, self.task.nh), -1e12)

        # Speed up tgt 1: clip it between hmin and hmax.
        bh_VhT = jnp.clip(bh_VhT, self.task.h_min, self.task.h_max)
        # Speed up tgt 2: If its smaller than h, then make it h.
        bh_tgt = jax_vmap(self.task.h_components)(data.b_xT)
        bh_VhT = jnp.maximum(bh_VhT, bh_tgt)

        # 1.2: Compute (target) V at T_x using b_vterms
        bh_rhs = data.bh_int_rhs + data.b_discount_rhs[:, None] * bh_VhT
        bh_V_tgt = data.bh_lhs + self.tgt_rhs_coeff * jnp.maximum(0, bh_rhs - data.bh_lhs)
        assert bh_V_tgt.shape == (b_size, self.task.nh)

        Vh_new, info_h = self.update_Vh(loss_weights, data.b_x0, data.b_u0, data.bh_iseqh, bh_V_tgt)
        (f_new, G_new), info_f_G = self.update_f_G(data.b_x0, data.b_u0, data.b_nxt_x0)
        info_mean = info_h | info_f_G

        info_mean["anneal/lam"] = self.lam
        gamma = jnp.exp(-self.lam * self.task.dt)
        info_mean["anneal/eff_horizon"] = 1 / (1 - gamma)
        info_mean["anneal/tgt_rhs_coeff"] = self.tgt_rhs_coeff
        info_mean["collect_idx"] = self.collect_idx
        info_mean["update_idx"] = self.update_idx

        Vh_tgt_params = optax.incremental_update(Vh_new.params, self.Vh_tgt.params, self.train_cfg.tau)
        Vh_tgt_new = self.Vh_tgt.replace(params=Vh_tgt_params)

        return self.replace(Vh = Vh_new, f = f_new, G = G_new, Vh_tgt=Vh_tgt_new, update_idx=self.update_idx + 1), info_mean

    def get_cbf_qpmats(self, alpha_safe: float, alpha_unsafe: float, state: State, V_shift: float = 1e-3, nom_pol=None):
        u_lb, u_ub = self.task.u_min, self.task.u_max
        Vh_apply = ft.partial(self.get_Vh, params=self.Vh.params)
        nom_pol = get_or(nom_pol, self.nom_pol)

        h_V = Vh_apply(state)
        hx_Vx = jax.jacobian(Vh_apply)(state)
        f = self.get_f(state)
        G = self.get_G(state)
        u_nom = nom_pol(state)

        # Give a small margin.
        h_V = h_V + V_shift

        if isinstance(alpha_safe, float) or alpha_safe.ndim == 0:
            is_safe = jnp.all(h_V < 0)
            alpha = jnp.where(is_safe, alpha_safe, alpha_unsafe)
        else:
            assert alpha_safe.shape == alpha_unsafe.shape == (self.task.nh,)
            h_is_safe = h_V < 0
            h_alpha_safe, h_alpha_unsafe = alpha_safe, alpha_unsafe
            h_alpha = jnp.where(h_is_safe, h_alpha_safe, h_alpha_unsafe)
            alpha = h_alpha

        penalty, relax_eps1, relax_eps2 = 10.0, 5e-1, 20.0
        qp = min_norm_cbf_qp_mats(alpha, u_lb, u_ub, h_V, hx_Vx, f, G, u_nom, penalty, relax_eps1, relax_eps2)
        return qp

    def get_cbf_control_sloped_all(
        self, alpha_safe: float, alpha_unsafe: float, state: State, V_shift: float = 1e-3, nom_pol=None
    ):
        u_lb, u_ub = self.task.u_min, self.task.u_max
        Vh_apply = ft.partial(self.get_Vh, params=self.Vh.params)

        nom_pol = get_or(nom_pol, self.nom_pol)

        h_V = Vh_apply(state)
        h_Vx = jax.jacobian(Vh_apply)(state)
        f = self.get_f(state)
        G = self.get_G(state)
        u_nom = nom_pol(state)

        # Give a small margin.
        h_V = h_V + V_shift

        if isinstance(alpha_safe, float) or alpha_safe.ndim == 0:
            is_safe = jnp.all(h_V < 0)
            alpha = jnp.where(is_safe, alpha_safe, alpha_unsafe)
        else:
            assert alpha_safe.shape == alpha_unsafe.shape == (self.task.nh,)
            h_is_safe = h_V < 0
            h_alpha_safe, h_alpha_unsafe = alpha_safe, alpha_unsafe
            h_alpha = jnp.where(h_is_safe, h_alpha_safe, h_alpha_unsafe)
            alpha = h_alpha

        # u, r, (qp_state, qp_mats) = cbf_old.min_norm_cbf(alpha, u_lb, u_ub, h_V, h_Vx, f, G, u_nom)
        u, r, sol = min_norm_cbf(alpha, u_lb, u_ub, h_V, h_Vx, f, G, u_nom)
        return self.task.chk_u(u), (r, sol)
 
  

    @jax.jit
    def eval(self) -> IntAvoid.EvalData:
        # Get states for plotting and for metrics.
        b_x0_plot = self.task.get_plot_x0(0)
        b_x0_metric = self.task.get_metric_x0()
        # b_x0_loss = self.task.get_loss_x0()

        # Rollout using the min norm controller.
        alpha_safe, alpha_unsafe = 2.0, 100.0
        pol = ft.partial(self.get_cbf_control_sloped, alpha_safe, alpha_unsafe)
        # sim = SimNCLF(self.task, pol, self.cfg.eval_cfg.eval_rollout_T)
        eval_rollout_T = self.cfg.eval_cfg.eval_rollout_T
        tf = self.task.dt * eval_rollout_T

        # sim = SimCtsReal(self.task, pol, tf, self.task.dt, use_obs=False)
        # bT_x_plot, _, _ = jax_vmap(sim.rollout_plot)(b_x0_plot)
        # bT_x_metric, _, _ = jax_vmap(sim.rollout_plot)(b_x0_metric)

        # Don't use PID stepsize controller, since the QP controller is probably super nonsmooth.
        sim = SimCtsReal(self.task, pol, tf, self.task.dt, use_obs=False, use_pid=False, max_steps=eval_rollout_T + 1)
        bT_x_plot, _, _ = jax_vmap(sim.rollout_plot)(b_x0_plot)
        bT_x_metric, _, _ = jax_vmap(sim.rollout_plot)(b_x0_metric)

        def get_V_info(state):
            Vh_apply = ft.partial(self.get_Vh, params=self.Vh.params)
            h_V = Vh_apply(state)
            h_Vx = jax.jacobian(Vh_apply)(state)
            f, G = self.get_f(state), self.get_G(state)
            h_h = self.task.h_components(state)

            u_nom = self.nom_pol(state)
            is_safe = jnp.all(h_V < 0)
            alpha = jnp.where(is_safe, alpha_safe, alpha_unsafe)
            # u, _, _ = cbf_old.min_norm_cbf(alpha, u_lb, u_ub, h_V, h_Vx, f, G, u_nom)
            u, _, _ = min_norm_cbf(alpha, u_lb, u_ub, h_V, h_Vx, f, G, u_nom)
            xdot = f + jnp.sum(G * u, axis=-1)
            h_Vdot = jnp.sum(h_Vx * xdot, axis=-1)

            h_Vdot_disc = h_Vdot - self.lam * (h_V - h_h)

            return h_V, h_Vdot, h_Vdot_disc, u

        # def get_loss_info(state):
        #     Vh_apply = ft.partial(self.get_Vh, params=self.Vh.params)
        #     h_V = Vh_apply(state)
        #     h_Vx = jax.jacobian(Vh_apply)(state)
        #     h_h = self.task.h_components(state)
        #
        #     # Check descent condition along pi.
        #     f, G = self.task.f(state), self.task.G(state)
        #     u_nom = self.nom_pol(state)
        #     # is_safe = jnp.all(h_V < 0)
        #     # alpha = jnp.where(is_safe, alpha_safe, alpha_unsafe)
        #     # u_qp, _, _ = cbf_old.min_norm_cbf(alpha, u_lb, u_ub, h_V, h_Vx, f, G, u_nom)
        #     # u_qp, _, _ = min_norm_cbf(alpha, u_lb, u_ub, h_V, h_Vx, f, G, u_nom)
        #
        #     xdot_nom = f + jnp.sum(G * u_nom, axis=-1)
        #     # xdot_qp = f + jnp.sum(G * u_qp, axis=-1)
        #
        #     h_Vdot_nom = jnp.sum(h_Vx * xdot_nom, axis=-1)
        #     # h_Vdot_qp = jnp.sum(h_Vx * xdot_qp, axis=-1)
        #
        #     h_Vdot_disc_nom = h_Vdot_nom - self.lam * (h_V - h_h)
        #
        #     # h_h > 0  =>  h_V > 0
        #     # equiv as  ~( h_h > 0 ) or (h_V > 0)
        #     h_clsfy = (h_h <= 0) | (h_V > 0)
        #
        #     return h_clsfy, h_V - h_h, h_Vdot_nom, h_Vdot_disc_nom

        u_lb, u_ub = self.task.u_min, self.task.u_max
        bb_x, bb_Xs, bb_Ys = self.task.get_contour_x0()
        bbh_V, bbh_Vdot, bbh_Vdot_disc, bb_u = rep_vmap(get_V_info, rep=2)(bb_x)

        # bh_clsfy, bh_hmV, bh_Vdot_nom, bh_Vdot_disc_nom = jax_vmap(get_loss_info)(b_x0_loss)
        # h_desc_nom = jnp.mean(bh_Vdot_nom <= 0, axis=0)
        # h_desc_disc_nom = jnp.mean(bh_Vdot_disc_nom <= 0, axis=0)
        # # h_desc_qp = jnp.mean(bh_Vdot_qp <= 0, axis=0)
        # desc_nom_max = (bh_Vdot_nom.max(axis=1) <= 0).mean()
        # desc_nom_disc_max = (bh_Vdot_disc_nom.max(axis=1) <= 0).mean()
        # # desc_qp_max = (bh_Vdot_qp.max(axis=1) <= 0).mean()
        # h_hmV_neg = jnp.mean(bh_hmV <= 0, axis=0)
        # hmV_neg_max = jnp.mean(bh_hmV.max(axis=1) <= 0, axis=0)
        #
        # h_clsfy = bh_clsfy.mean(axis=0)
        # h_clsfy_max = jnp.all(bh_clsfy, axis=1).mean()

        loss_info = {}
        # for ii, h_label in enumerate(self.task.h_labels_clean):
        #     loss_info[f"Descent/Nom/{h_label}"] = h_desc_nom[ii]
        #     loss_info[f"Descent/Nom Disc/{h_label}"] = h_desc_disc_nom[ii]
        #     # loss_info[f"Descent/QP/{h_label}"] = h_desc_qp[ii]
        #     loss_info[f"ClassifyDet/{h_label}"] = h_hmV_neg[ii]
        #     loss_info[f"Classify/{h_label}"] = h_clsfy[ii]
        # loss_info[f"Descent/Nom/Max"] = desc_nom_max
        # loss_info[f"Descent/Nom Disc/Max"] = desc_nom_disc_max
        # loss_info[f"Descent/QP/Max"] = desc_qp_max
        # loss_info[f"ClassifyDet/Max"] = hmV_neg_max
        # loss_info[f"Classify/Max"] = h_clsfy_max

        # bTl_ls = rep_vmap(self.task.l_components, rep=2)(bT_x_metric)
        bTh_hs = rep_vmap(self.task.h_components, rep=2)(bT_x_metric)
        bT_hs = bTh_hs.max(axis=2)

        h_mean = bT_hs.mean(-1).mean(0)
        h_max = bT_hs.max(-1).mean(0)

        h_labels = self.task.h_labels_clean
        safe_fracs = {
            "Safe/{}".format(h_labels[ii]): jnp.all(bTh_hs[:, :, ii] < 0, axis=1).mean() for ii in range(self.task.nh)
        }

        eval_info = {
            "Constr Mean": h_mean,
            "Constr Max Mean": h_max,
            "Safe Frac": jnp.mean(jnp.all(bT_hs < 0, axis=-1)),
            **safe_fracs,
            **loss_info,
        }

        return self.EvalData(bT_x_plot, bb_Xs, bb_Ys, bbh_V, bbh_Vdot, bbh_Vdot_disc, bb_u, eval_info)

 
 
class GumbelModelBasedIntAvoid(ModelBasedIntAvoid):
    
    @classmethod
    def create_from_base(cls, baseline: ModelBasedIntAvoid) -> "GumbelModelBasedIntAvoid":
        base_kwargs = {k: getattr(baseline, k) for k in baseline.__dataclass_fields__}
        return cls(**base_kwargs)
        

    def get_fs(self, state: State, params=None): #,normalizer: Normalizer):
        params = get_or(params, self.f.params)
        #normalized_state = normalizer.transform_state(state) 
        #assert normalized_state.shape == (self.task.nx,)
        f = self.f.apply_fn(params, state) #normalized_state(state))
        assert f.shape == (self.cfg.n_fs, self.task.nx)
        return f 
 
    def get_Gs(self, state: State, params=None): #,normalizer: Normalizer):
        params = get_or(params, self.G.params)
        #normalized_state = normalizer.transform_state(state)  

        #assert normalized_state.shape == (self.task.nx, ) 
        G = self.G.apply_fn(params, state) #normalized_state(state))
        assert G.shape == (self.cfg.n_Gs, self.task.nx * self.task.nu)
        return G.reshape(self.cfg.n_Gs, self.task.nx, self.task.nu)
    
    
    def compute_loss(self, loss_weights, b_x: BState, b_u: BControl, bh_iseqh: BHBool, bh_V_tgt: BHFloat, params):
        """
        Compute the loss for the given batch of states and controls.
        Parameters:
        -----------
        loss_weights : dict
            Dictionary containing the weights for different loss components.
        b_x : BState
            Batch of states.
        b_u : BControl
            Batch of controls.
        bh_iseqh : BHBool
            Boolean array indicating whether V equals h.
        bh_V_tgt : BHFloat
            Target value function.
        params : dict
            Additional parameters for the loss computation.
        Returns:
        --------
        loss : jnp.ndarray
            The computed loss value.
        loss_dict : dict
            Dictionary containing individual loss components and additional information.
        """
        batch_size = len(b_x)
        eh_V_apply = ft.partial(self.get_eh_Vh, params=params)

        beh_V_pred = jax_vmap(eh_V_apply)(b_x)
        # behx_Vx = jax_vmap(jax.jacobian(eh_V_apply))(b_x)
        bh_h = jax_vmap(self.task.h_components)(b_x)

        # 1: Value function loss. This applies to all states in the trajectory.
        eh_loss_Vh = jnp.mean((bh_V_tgt[:, None, :] - beh_V_pred) ** 2, axis=0)
        assert eh_loss_Vh.shape == (self.cfg.n_Vs, self.task.nh)
        loss_Vh = jnp.mean(eh_loss_Vh)

        # Penalize underestimating beh_Vpred. i.e., Penalize tgt >= pred.
        eh_loss_Vh_tilt = jnp.mean(jnn.relu(bh_V_tgt[:, None, :] - beh_V_pred) ** 2, axis=0)
        loss_Vh_tilt = jnp.mean(eh_loss_Vh_tilt)


        # 2: Descent loss.
        b_fs = jax_vmap(self.get_fs)(b_x)
        b_Gs = jax_vmap(self.get_Gs)(b_x)

        def compute_terms_h(h_V, x: State, h_h, fs, Gs, u):
            '''
            ## Original
            xdot = lax.stop_gradient(fs.mean(axis = 0) + Gs.mean(axis = 0) @ u) 
            _, eh_Vdot = jax.jvp(eh_V_apply, (x,), (xdot,))
            assert eh_Vdot.shape == (self.cfg.n_Vs, self.task.nh)
            future = eh_Vdot - self.lam * (h_V - h_h) 
            return future
            '''

            ### Quantile robust 
            xdots = lax.stop_gradient(jax_vmap(lambda f, G: f + G @ u)(fs, Gs))
            # Compute Vx^T xdot
            _, eh_Vdots = jax_vmap(lambda xdot: jax.jvp(eh_V_apply, (x,), (xdot,)))(xdots)
            assert eh_Vdots.shape == (self.cfg.n_fs, self.cfg.n_Vs, self.task.nh)
            futures = eh_Vdots - self.lam * (h_V - h_h)[None, ...]
            def quantile(futures_single_v):
                #sorted_futures_single_v = jnp.sort(futures_single_v, axis = 0)
                fugures_single_v_quantile = jnp.quantile(futures_single_v, 0.8, axis = 0)
                filtered_fugures_single_v = jnp.where(futures_single_v <= fugures_single_v_quantile, futures_single_v, 0)
                return filtered_fugures_single_v.max(axis = 0)
            
            future = jax.vmap(quantile, in_axes = 1)(futures)
            return future
            

        grad_terms = {}
        info_dict = {}

        
        if self.train_cfg.use_grad_terms:
            beh_now = bh_h[:, None, :] - beh_V_pred
            beh_future = jax_vmap(compute_terms_h)(beh_V_pred, b_x, bh_h, b_fs, b_Gs, b_u)
 
            assert beh_now.shape == beh_future.shape == (batch_size, self.cfg.n_Vs, self.task.nh)
            # Enforce h(x) - V(x) <= 0.
            loss_now_pos = jnp.mean(jnn.relu(beh_now) ** 2)
            acc_now = jnp.mean(beh_now <= 0)

            # Enforce Vdot - lambda * (V - h) <=0.
            loss_future_pos = jnp.mean(jnn.relu(beh_future) ** 2)
            acc_future = jnp.mean(beh_future <= 0)

            # Represent the complementarity as a multiplication.
            # If it's positive, clip it to force it to be non-positive.
            loss_pde = jnp.mean((beh_now * beh_future) ** 2)

            grad_terms = {
                "Loss/Now": loss_now_pos,
                "Loss/Future": loss_future_pos,
                "Loss/PDE": loss_pde,
            }
            info_dict = {"Acc/V_desc": acc_now, "Acc/Nonzero": acc_future}
        else:
            if ("Loss/Future" in loss_weights) or ("Loss/Now" in loss_weights):
                raise ValueError("Set use_grad_terms to True to use Loss/Future or Loss/Now!")

        # 3: Gradient loss for when V = h.
        if self.train_cfg.use_hgrad:
            bh_V_pred = lax.stop_gradient(jnp.mean(beh_V_pred, axis=1))
            bh_iseqh = bh_iseqh & (bh_V_pred <= bh_V_tgt)

            b_xdot = b_fs.mean(axis = 1) + jnp.sum(b_Gs.mean(axis = 1) * b_u[:, None, :], axis=-1)

            # bh_hdot = jnp.sum(bhx_hgrad * b_xdot, axis=-1)
            _, bh_hdot = jax.vmap(lambda x, xdot: jax.jvp(self.task.h_components, (x,), (xdot,)))(b_x, b_xdot)
            assert bh_hdot.shape == (batch_size, self.task.nh)

            _, beh_Vdot = jax.vmap(lambda x, xdot: jax.jvp(eh_V_apply, (x,), (xdot,)))(b_x, b_xdot)
            assert beh_Vdot.shape == (batch_size, self.cfg.n_Vs, self.task.nh)

            ### Original: let Vdot==hdot at states where V = h
            bh_loss_dV_eqh = jnp.mean((beh_Vdot - bh_hdot[:, :, None, :]) ** 2, axis=1)

            ### Majorization: penalize hdot >= Vdot so that h decreases faster than V 
            #bh_loss_dV_eqh += jnn.relu(jnp.exp(10. * (bh_hdot[:, :, None, :] - beh_Vdot)) - 1.0).mean(axis = 1)
            
            assert bh_loss_dV_eqh.shape == (batch_size, self.task.nh)

            ### Original: let Vdot be zero at states where V > h because V equals future maximum h
            bh_loss_dV_eq0 = jnp.mean(beh_Vdot**2, axis=1)

            ### Majorization: heavily penalize Vdot <= 0 so that V does not decrease
            #bh_loss_dV_eq0 += jnn.relu(jnp.exp( - 10. * beh_Vdot) - 1.0).mean(axis = 1)

            assert bh_loss_dV_eq0.shape == (batch_size, self.task.nh)

            bh_loss_dV = jnp.where(bh_iseqh, bh_loss_dV_eqh, bh_loss_dV_eq0)
            loss_dV = jnp.mean(bh_loss_dV)

            grad_terms = {"Loss/dV_mse": loss_dV}

        extra_loss = {}
        if self.train_cfg.use_eq_state:
            assert self.task.has_eq_state()
            x_eq = self.task.eq_state()
            eh_V_pred_eq = eh_V_apply(x_eq)
            h_eq = self.task.h_components(x_eq)
            loss_Vh_eq = jnp.mean((h_eq - eh_V_pred_eq) ** 2)
            extra_loss["Loss/Equil"] = loss_Vh_eq

        loss_dict = {
            "Loss/Vh_mse": loss_Vh,
            "Loss/Vh_tilt": loss_Vh_tilt,
            **grad_terms,
            # "Loss/Hess Fro": loss_fro,
            **extra_loss,
        }
        loss = weighted_sum_dict(loss_dict, loss_weights)
        return loss, loss_dict | info_dict
    