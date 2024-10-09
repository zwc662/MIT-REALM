from typing import NamedTuple

import einops as ei
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import optax
from attrs import define
from flax import struct

import clfrl.utils.typed_ft as ft
from clfrl.dyn.dyn_types import BBLFloat, BControl, BLFloat, BPolObs, BTEFloat, BTEState, BTState, BVObs, LFloat
from clfrl.dyn.task import Task
from clfrl.networks.ensemble import Ensemble, subsample_ensemble
from clfrl.networks.mlp import MLP
from clfrl.networks.network_utils import HidSizes, get_act_from_str
from clfrl.networks.optim import get_default_tx
from clfrl.networks.policy_net import TanhNormal
from clfrl.networks.q_net import QValueNet
from clfrl.networks.temperature import Temperature
from clfrl.networks.train_state import TrainState
from clfrl.rl.collector import Collector, CollectorCfg, CollectState, Experience
from clfrl.rl.replay_buffer_np import ReplayBufferNp
from clfrl.rl.sim_rl import SimMode, SimRng
from clfrl.utils.grad_utils import empty_grad_tx
from clfrl.utils.jax_types import BBFloat, BFloat, FloatScalar, MetricsDict
from clfrl.utils.jax_utils import jax_vmap, merge01, rep_vmap, tree_copy, tree_split_dims
from clfrl.utils.none import get_or
from clfrl.utils.rng import PRNGKey
from clfrl.utils.schedules import Schedule
from clfrl.utils.shape_utils import assert_scalar, assert_shape
from clfrl.utils.tfp import tfd


@define
class SACTrainCfg:
    disc_gamma: float
    # Target Q-function smoothing. Small values = more smoothing.
    tau: float

    # Batch size to use when updating.
    batch_size: int
    # Number of val updates per policy update.
    val_pol_ratio: int
    # Number of val updates per (batch of) environment interaction.
    val_act_ratio: int
    # If None, use -nu / 2.
    target_entropy: float | None


@define
class SACEvalCfg:
    # How long to rollout during eval.
    eval_rollout_T: int


@define
class SACCfg:
    n_qs: int
    n_min_qs: int

    act: str
    pol_lr: Schedule
    val_lr: Schedule
    temp_lr: Schedule

    pol_hids: HidSizes
    val_hids: HidSizes

    init_temp: float
    critic_layernorm: bool

    collect_cfg: CollectorCfg
    train_cfg: SACTrainCfg
    eval_cfg: SACEvalCfg

    @property
    def alg_name(self) -> str:
        return "SAC"


class SACAlg(struct.PyTreeNode):
    collect_idx: int
    key: PRNGKey
    policy: TrainState[tfd.Distribution]
    Ql: TrainState[LFloat]
    Ql_tgt: TrainState[LFloat]
    temp: TrainState[FloatScalar]

    cfg: SACCfg = struct.field(pytree_node=False)
    task: Task = struct.field(pytree_node=False)

    class Batch(NamedTuple):
        b_Vobs: BVObs
        b_polobs: BPolObs
        b_control: BControl
        b_Vobs_next: BVObs
        b_polobs_next: BPolObs
        bl_l: BLFloat

    class EvalData(NamedTuple):
        bT_x_plot: BTState
        bTe_Vl: BTEFloat
        bTe_Vl_dot: BTEFloat
        bT_x_plot_rng: BTState

        bb_Xs: BBFloat
        bb_Ys: BBFloat
        bbl_Vl_mean: BBLFloat

        info: MetricsDict

    @classmethod
    def create(cls, seed: int, task: Task, cfg: SACCfg) -> "SACAlg":
        key, key_pol, key_Ql, key_temp = jr.split(jr.PRNGKey(seed), 4)

        pol_obs, V_obs, action = np.zeros(task.n_polobs), np.zeros(task.n_Vobs), np.zeros(task.nu)

        act = get_act_from_str(cfg.act)

        # Define policy network.
        pol_base_cls = ft.partial(MLP, cfg.pol_hids, act)
        pol_def = TanhNormal(pol_base_cls, task.nu)
        pol_tx = get_default_tx(cfg.pol_lr.make())
        policy = TrainState.create_from_def(key_pol, pol_def, (pol_obs,), tx=pol_tx)

        # Define Ql network.
        Ql_base_cls = ft.partial(MLP, cfg.val_hids, act, cfg.critic_layernorm)
        Ql_cls = ft.partial(QValueNet, Ql_base_cls, task.nl)
        Ql_def = Ensemble(Ql_cls, num=cfg.n_qs)
        Ql_tx = get_default_tx(cfg.val_lr.make())
        Ql = TrainState.create_from_def(key_Ql, Ql_def, (pol_obs, action), Ql_tx)

        Ql_tgt_def = Ensemble(Ql_cls, num=cfg.n_min_qs)
        Ql_params_copy = tree_copy(Ql.params)  # Copy so that we can donate buffer.
        Ql_tgt = TrainState.create(apply_fn=Ql_tgt_def.apply, params=Ql_params_copy, tx=empty_grad_tx())

        temp_def = Temperature(cfg.init_temp)
        temp_tx = get_default_tx(cfg.temp_lr.make())
        temp = TrainState.create_from_def(key_temp, temp_def, tuple(), temp_tx)

        cfg.train_cfg.target_entropy = get_or(cfg.train_cfg.target_entropy, -task.nu / 2)

        return SACAlg(jnp.array(0, dtype=jnp.int32), key, policy, Ql, Ql_tgt, temp, cfg, task)

    @property
    def train_cfg(self):
        return self.cfg.train_cfg

    @property
    def epoch_batch_size(self) -> int:
        return self.cfg.collect_cfg.collect_batch * self.train_cfg.val_act_ratio

    @property
    def n_qs(self):
        return self.cfg.n_qs

    @property
    def n_min_qs(self):
        return self.cfg.n_min_qs

    def update_pol(self, batch: Batch) -> tuple["SACAlg", MetricsDict]:
        def get_loss(pol_params) -> tuple[FloatScalar, MetricsDict]:
            dist = self.policy.apply_fn(pol_params, batch.b_polobs)
            b_control = dist.sample(seed=key)
            b_logprobs = assert_shape(dist.log_prob(b_control), batch_size)

            ebl_Ql = assert_shape(self.Ql.apply(batch.b_polobs, b_control), (self.n_qs, batch_size, self.task.nl))
            eb_Ql = ebl_Ql.sum(axis=-1)
            b_Ql = eb_Ql.mean(axis=0)
            Ql = b_Ql.mean(axis=0)

            temperature = assert_scalar(self.temp.apply())

            mean_logprobs = b_logprobs.mean()
            pol_loss = assert_scalar(mean_logprobs * temperature + Ql)
            return pol_loss, {"pol_loss": pol_loss, "entropy": -mean_logprobs}

        batch_size = len(batch.b_polobs)
        key, key_self = jr.split(self.key, 2)
        grads, pol_info = jax.grad(get_loss, has_aux=True)(self.policy.params)
        policy = self.policy.apply_gradients(grads)
        return self.replace(key=key_self, policy=policy), pol_info

    def update_temperature(self, entropy: FloatScalar) -> tuple["SACAlg", MetricsDict]:
        def get_temp_loss(temp_params) -> tuple[FloatScalar, MetricsDict]:
            temperature = assert_scalar(self.temp.apply_fn(temp_params))
            assert_scalar(entropy)
            entropy_diff = assert_scalar(entropy - self.train_cfg.target_entropy)
            temp_loss = assert_scalar(temperature * entropy_diff)
            return temp_loss, {"temperature": temperature, "temperature_loss": temp_loss}

        grads, info = jax.grad(get_temp_loss, has_aux=True)(self.temp.params)
        temp = self.temp.apply_gradients(grads)
        return self.replace(temp=temp), info

    def update_Ql(self, batch: Batch) -> tuple["SACAlg", MetricsDict]:
        def get_Ql_loss(Vl_params) -> tuple[FloatScalar, MetricsDict]:
            ebl_Ql_pred = assert_shape(
                self.Ql.apply_fn(Vl_params, batch.b_Vobs, batch.b_control), (self.n_qs, batch_size, self.task.nl)
            )
            ebl_Ql_loss = (ebl_Ql_pred - bl_Ql_tgt) ** 2
            assert_shape(ebl_Ql_loss, (self.n_qs, batch_size, self.task.nl))
            Ql_loss = ebl_Ql_loss.mean()
            eb_Ql_pred = ebl_Ql_pred.sum(-1)
            return Ql_loss, {"Ql_loss": Ql_loss, "Ql": eb_Ql_pred.mean()}

        batch_size, num_qs, num_min_qs = len(batch.b_polobs), self.n_qs, self.n_min_qs
        b_Vobs_next = batch.b_Vobs_next

        dist_next = self.policy.apply(batch.b_polobs_next)
        key_pol, key_redq, key_self = jr.split(self.key, 3)
        b_u_next = dist_next.sample(seed=key_pol)
        target_params = subsample_ensemble(key_redq, self.Ql_tgt.params, num_min_qs, num_qs)
        ebl_Ql_next = self.Ql_tgt.apply_fn(target_params, b_Vobs_next, b_u_next)
        assert_shape(ebl_Ql_next, (num_min_qs, batch_size, self.task.nl))
        bl_Ql_next = assert_shape(ebl_Ql_next.max(axis=0), (batch_size, self.task.nl))

        bl_Ql_tgt = assert_shape(batch.bl_l + self.train_cfg.disc_gamma * bl_Ql_next, (batch_size, self.task.nl))

        grads, info = jax.grad(get_Ql_loss, has_aux=True)(self.Ql.params)
        Ql = self.Ql.apply_gradients(grads)

        Ql_tgt_params = optax.incremental_update(Ql.params, self.Ql.params, self.train_cfg.tau)
        Ql_tgt = self.Ql_tgt.replace(params=Ql_tgt_params)

        return self.replace(key=key_self, Ql=Ql, Ql_tgt=Ql_tgt), info

    @ft.partial(jax.jit, donate_argnums=0)
    def update(self, batch: Batch) -> tuple["SACAlg", MetricsDict]:
        val_act_ratio, val_pol_ratio = self.train_cfg.val_act_ratio, self.train_cfg.val_pol_ratio

        # Reshape batch.
        dataset_size = len(batch.b_polobs)
        if not (dataset_size % val_act_ratio == 0):
            raise ValueError(f"dataset size should be a multiple of the val_act_ratio. {dataset_size} {val_act_ratio}")
        if not (val_act_ratio % val_pol_ratio == 0):
            raise ValueError(f"val_act_ratio should be a multiple of val_pol_ratio. {val_act_ratio} {val_pol_ratio}")

        n_pol_updates = val_act_ratio // val_pol_ratio
        batch_size = dataset_size // val_act_ratio
        batch = tree_split_dims(batch, (n_pol_updates, val_pol_ratio, batch_size))

        def updates_body_outer(alg_outer: SACAlg, minibatch_outer: SACAlg.Batch):
            """Inner loop updates value function val_pol_ratio times and policy 1 time."""

            def updates_body(alg_: SACAlg, minibatch: SACAlg.Batch):
                alg_, Ql_info_ = alg_.update_Ql(minibatch)
                return alg_, Ql_info_

            alg, Ql_infos = lax.scan(updates_body, alg_outer, minibatch_outer, length=val_pol_ratio)
            Ql_info = jtu.tree_map(lambda x: jnp.mean(x, axis=0), Ql_infos)

            pol_minibatch = jtu.tree_map(lambda x: x[-1], minibatch_outer)
            alg, pol_info = alg.update_pol(pol_minibatch)
            alg, temp_info = alg.update_temperature(pol_info["entropy"])

            return alg, {**pol_info, **Ql_info, **temp_info}

        alg, b_info = lax.scan(updates_body_outer, self, batch, length=n_pol_updates)
        info = jtu.tree_map(lambda x: jnp.mean(x, axis=0), b_info)

        return alg, info

    def init_rb(self, rb_capacity: int) -> ReplayBufferNp:
        s = self.task
        pol_obs, V_obs = np.zeros(s.n_polobs, dtype=jnp.float32), jnp.zeros(s.n_Vobs, dtype=jnp.float32)
        control, l = np.zeros(s.nu, dtype=jnp.float32), jnp.zeros(s.nl, dtype=jnp.float32)
        dummy_sac_batch = SACAlg.Batch(pol_obs, V_obs, control, pol_obs, V_obs, l)
        return ReplayBufferNp.create(dummy_sac_batch, rb_capacity)

    @property
    def collector(self):
        return Collector(self.task, self.cfg.collect_cfg)

    @ft.partial(jax.jit, donate_argnums=0)
    def init_collect(self) -> tuple["SACAlg", CollectState]:
        key, key_self = jr.split(self.key, 2)
        return self.replace(key=key_self), self.collector.init_collect(key)

    @ft.partial(jax.jit, donate_argnums=(0, 1))
    def collect_data(self, collect_state: CollectState) -> tuple["SACAlg", CollectState, Batch]:
        key_collect, key_self = jr.split(self.key, 2)
        collect_state_new, bT_exp = self.collector.collect(key_collect, collect_state, self.policy.apply)

        # Flatten out bT_exp.
        bT_Vobs_i, bT_polobs_i = bT_exp.bTp1_Vobs[:, :-1], bT_exp.bTp1_polobs[:, :-1]
        bT_Vobs_n, bT_polobs_n = bT_exp.bTp1_Vobs[:, 1:], bT_exp.bTp1_polobs[:, 1:]
        bT_batch = self.Batch(bT_Vobs_i, bT_polobs_i, bT_exp.bT_u, bT_Vobs_n, bT_polobs_n, bT_exp.bTl_l)
        # TODO: n_step returns?
        b_batch = jtu.tree_map(lambda x: merge01(x), bT_batch)

        return self.replace(collect_idx=self.collect_idx + 1, key=key_self), collect_state_new, b_batch

    @jax.jit
    def eval(self) -> EvalData:
        # Get states for plotting and for metrics.
        b_x0_plot = self.task.get_plot_x0()
        b_x0_metric = self.task.get_metric_x0()
        b_x0_rng = self.task.get_plot_rng_x0()

        # Rollout.
        sim = SimMode(self.task, self.policy.apply, self.cfg.eval_cfg.eval_rollout_T)
        bT_x_plot, _ = jax_vmap(sim.rollout_plot)(b_x0_plot)
        bT_x_metric, _ = jax_vmap(sim.rollout)(b_x0_metric)

        n_plot_rng = 64
        b_x0_rng = merge01(ei.repeat(b_x0_rng, "b nx -> b rep nx", rep=n_plot_rng // len(b_x0_rng)))
        n_plot_rng = len(b_x0_rng)
        b_keys = jr.split(jr.PRNGKey(58123), n_plot_rng)
        sim = SimRng(self.task, self.policy.apply, self.cfg.eval_cfg.eval_rollout_T)
        bT_x_rng, _ = jax_vmap(ft.partial(sim.rollout))(b_keys, b_x0_rng)

        def get_e_Vl(state):
            Vobs, polobs = self.task.get_obs(state)
            dist = self.policy.apply(polobs)
            control_mode = dist.mode()
            el_Vl = self.Ql.apply(Vobs, control_mode)
            e_Vl = el_Vl.sum(-1)
            return e_Vl

        def get_e_Vl_and_dot(state):
            e_Vl = get_e_Vl(state)
            e_Vl_grad = jax.jacobian(get_e_Vl)(state)

            Vobs, polobs = self.task.get_obs(state)
            dist = self.policy.apply(polobs)
            control_mode = dist.mode()
            xdot = self.task.xdot(state, control_mode)

            e_Vl_dot = jnp.sum(e_Vl_grad * xdot, axis=-1)

            return e_Vl, e_Vl_dot

        rnd_idxs = np.random.default_rng(seed=314159)
        V_idxs = rnd_idxs.choice(len(bT_x_plot), size=8, replace=False)
        bTe_Vl, bTe_Vl_dot = rep_vmap(get_e_Vl_and_dot, rep=2)(bT_x_plot[V_idxs])

        # Get states on grid for plotting value function / policy.
        def get_Vl(state):
            Vobs, polobs = self.task.get_obs(state)
            dist = self.policy.apply(polobs)
            control_mode = dist.mode()
            el_Vl = self.Ql.apply(Vobs, control_mode)
            l_Vl_mean = el_Vl.mean(0)
            return control_mode, l_Vl_mean

        bb_x, bb_Xs, bb_Ys = self.task.get_contour_x0()
        bb_pol, bbl_Vl_mean = rep_vmap(get_Vl, rep=2)(bb_x)

        bTl_ls = rep_vmap(self.task.l_components, rep=2)(bT_x_metric)

        l_mean = bTl_ls.sum(1).mean(0)
        l_final = bTl_ls[:, -1, :].mean(0)
        eval_info = {"Cost Mean": l_mean, "Cost Final Mean": l_final}

        return self.EvalData(bT_x_plot, bTe_Vl, bTe_Vl_dot, bT_x_rng, bb_Xs, bb_Ys, bbl_Vl_mean, eval_info)
