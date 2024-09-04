import functools as ft
from typing import NamedTuple

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import optax
from attrs import define
from flax import struct
from loguru import logger

from efppo.networks.ef_wrapper import EFWrapper, ZEncoder
from efppo.networks.mlp import MLP
from efppo.networks.network_utils import ActLiteral, HidSizes, get_act_from_str, get_default_tx
from efppo.networks.policy_net import DiscretePolicyNet
from efppo.networks.train_state import TrainState 
from efppo.networks.critic_net import DoubleDiscreteCriticNet
from efppo.rl.collector import Collector, RolloutOutput, collect_single_mode, collect_single_env_mode
from efppo.rl.gae_utils import compute_efocp_gae, compute_efocp_V
from efppo.task.dyn_types import BControl, BHFloat, BObs, HFloat, LFloat, ZBBControl, ZBBFloat, ZBTState, ZFloat
from efppo.task.task import Task
from efppo.utils.cfg_utils import Cfg
from efppo.utils.grad_utils import compute_norm_and_clip
from efppo.utils.jax_types import BFloat, FloatScalar
from efppo.utils.jax_utils import jax_vmap, merge01, tree_split_dims, tree_stack, jax2np
from efppo.utils.rng import PRNGKey
from efppo.utils.schedules import Schedule, as_schedule
from efppo.utils.tfp import tfd 
from efppo.utils.replay_buffer import Experience, ReplayBuffer
from efppo.utils.svgd import compute_kernel_gradient, compute_kernel_matrix, svgd_update

@define
class BaselineCfg(Cfg):
    @define
    class TrainCfg(Cfg):
        z_min: float
        z_max: float

        n_batches: int
        bc_ratio: float 

        clip_grad_pol: float
        clip_grad_V: float

         

    @define
    class EvalCfg(Cfg):
        ...

    @define
    class NetCfg(Cfg):
        pol_lr: Schedule
        val_lr: Schedule

        # Schedules are in units of collect_idx.
        entropy_cf: Schedule
        disc_gamma: Schedule

        act: ActLiteral
        pol_hids: HidSizes
        val_hids: HidSizes
  
        nz_enc: int
        z_mean: float
        z_scale: float

        act_final: bool = True
 
        n_critics: int = 2
       

    net: NetCfg
    train: TrainCfg
    eval: EvalCfg


class BaselineSAC(struct.PyTreeNode):
    update_idx: int
    key: PRNGKey 
    temp: FloatScalar
    policy: TrainState[tfd.Distribution]
    critic: TrainState[LFloat]
    target_critic: TrainState[HFloat]
    disc_gamma: FloatScalar

    task: Task = struct.field(pytree_node=False)
    cfg: BaselineCfg = struct.field(pytree_node=False)

    ent_cf_sched: optax.Schedule = struct.field(pytree_node=False)
    disc_gamma_sched: optax.Schedule = struct.field(pytree_node=False)

    Cfg = BaselineCfg

    class Batch(NamedTuple):
        b_obs: BObs 
        b_nxt_obs: BObs
        b_z: BFloat
        b_nxt_z: BFloat
        b_control: BControl
        b_logprob: BFloat
        b_l: BFloat 
        b_expert_control: BControl
        
        @property
        def batch_size(self) -> int:
            assert self.b_logprob.ndim == 2
            return self.b_logprob.shape[1]
        
        @property
        def num_batches(self) -> int:
            assert self.b_logprob.ndim == 2
            return self.b_logprob.shape[0]

    class EvalData(NamedTuple):
        z_zs: ZFloat
        zbb_pol: ZBBControl
        zbb_prob: ZBBFloat
        zbb_critic: ZBBFloat
        zbb_target_critic: ZBBFloat

        zbT_x: ZBTState

        info: dict[str, float]

    @classmethod
    def create(cls, key: jr.PRNGKey, task: Task, cfg: BaselineCfg):
        key, key_pol, key_critic, key_replay_buffer = jr.split(key, 4)

        obs, z, control = np.zeros(task.nobs), np.array(0.0), np.array(0.0)
        act = get_act_from_str(cfg.net.act)

        # Encoder for z. Params not shared.
        z_base_cls = ft.partial(ZEncoder, nz=cfg.net.nz_enc, z_mean=cfg.net.z_mean, z_scale=cfg.net.z_scale)
        
        # Define policy network.
        pol_base_cls = ft.partial(MLP, cfg.net.pol_hids, act, act_final=cfg.net.act_final, scale_final=1e-2)
        pol_cls = ft.partial(DiscretePolicyNet, pol_base_cls, task.n_actions)
        pol_def = EFWrapper(pol_cls, z_base_cls)
        pol_tx = get_default_tx(as_schedule(cfg.net.pol_lr).make())
        pol = TrainState.create_from_def(key_pol, pol_def, (obs, z), pol_tx)

        # Define critic network.
        print(f"Ensembled {cfg.net.n_critics} critic networks")
        critic_base_cls = ft.partial(MLP, cfg.net.val_hids, act)
        critic_cls = ft.partial(DoubleDiscreteCriticNet, critic_base_cls, task.n_actions, cfg.net.n_critics)
        critic_def = EFWrapper(critic_cls, z_base_cls)
        critic_tx = get_default_tx(as_schedule(cfg.net.val_lr).make())
        critic = TrainState.create_from_def(key_critic, critic_def, (obs, z), critic_tx)

        # Define target_critic network.
        target_critic_base_cls = ft.partial(MLP, cfg.net.val_hids, act)
        target_critic_cls = ft.partial(DoubleDiscreteCriticNet, target_critic_base_cls, task.n_actions, cfg.net.n_critics)
        target_critic_def = EFWrapper(target_critic_cls, z_base_cls)
        target_critic_tx = get_default_tx(as_schedule(cfg.net.val_lr).make())
        target_critic = TrainState.create_from_def(key_critic, target_critic_def, (obs, z), target_critic_tx)

        ent_cf = as_schedule(cfg.net.entropy_cf).make()
        disc_gamma_sched = as_schedule(cfg.net.disc_gamma).make()
        disc_gamma = disc_gamma_sched(0)
 
        return BaselineSAC(0, key, 1, pol, critic, target_critic, disc_gamma, task, cfg, ent_cf, disc_gamma_sched)

    @property
    def train_cfg(self) -> BaselineCfg.TrainCfg:
        return self.cfg.train

    @property
    def z_min(self):
        return self.train_cfg.z_min

    @property
    def z_max(self):
        return self.train_cfg.z_max

    @property
    def ent_cf(self):
        return self.ent_cf_sched(self.update_idx)

    def make_dset(self, replay_buffer: ReplayBuffer, data: RolloutOutput) -> Batch:
        new_replay_buffer = replay_buffer.insert(data)

        num_batches = self.train_cfg.n_batches
        batch_size = 256 
        new_replay_buffer, batch_data = new_replay_buffer.sample(num_batches, batch_size)
        b_batch = self.Batch(
            b_obs = batch_data.Tp1_obs,
            b_nxt_obs = batch_data.Tp1_nxt_obs,
            b_z = batch_data.Tp1_z,
            b_nxt_z = batch_data.Tp1_nxt_z,
            b_control = batch_data.T_control,
            b_logprob = batch_data.T_logprob,
            b_l = batch_data.T_l,
            b_expert_control = batch_data.T_expert_control
        )
        return new_replay_buffer, b_batch

    @ft.partial(jax.jit, donate_argnums=0)
    def update(self, batch: Batch) -> tuple["BaselineSAC", dict]:
        def update_one_batch(obj, batch_idx):
            key_shuffle, key_self = jr.split(obj.key, 2)
            rand_idxs = jr.permutation(key_shuffle, jnp.arange(batch.batch_size))
            mb_dset = jax.tree_map(lambda x: x[batch_idx][rand_idxs], batch)
            
            # 3: Perform value function and policy updates.
            def updates_body(alg_: BaselineSAC, b_batch: BaselineSAC.Batch):
                alg_, critic_info = alg_.update_critic(b_batch)
                alg_, pol_info = alg_.update_policy(b_batch)
                return alg_, critic_info | pol_info

            new_obj, info = lax.scan(updates_body, obj, mb_dset, length=batch.num_batches)
            # Take the mean.
            info = jax.tree_map(jnp.mean, info)

            info["steps/policy"] = obj.policy.step
            info["steps/critic"] = obj.critic.step
            info["anneal/ent_cf"] = obj.ent_cf

            return new_obj.replace(key=key_self, update_idx=obj.update_idx + 1), info
        
        new_self, new_info = lax.scan(update_one_batch, self, jnp.arange(batch.num_batches))
 
        return new_self, jtu.tree_map(jnp.mean, new_info)
    

    def update_iteratively(self, batch: Batch) -> tuple["BaselineSAC", dict]:
        info = {}
        for batch_idx in range(batch.num_batches):
            key_shuffle, key_self = jr.split(self.key, 2)
            rand_idxs = jr.permutation(key_shuffle, jnp.arange(batch.batch_size))
            mb_dset = jax.tree_map(lambda x: x[batch_idx][rand_idxs], batch)
            
            # 3: Perform value function and policy updates.
            def updates_body(alg_: BaselineSAC, b_batch: BaselineSAC.Batch):
                alg_, critic_info = alg_.update_critic(b_batch)
                alg_, pol_info = alg_.update_policy(b_batch)
                return alg_, critic_info | pol_info

            self, info = updates_body(self, mb_dset)
            # Take the mean.
            
            info = jax.tree_map(jnp.mean, info)

            info["steps/policy"] = self.policy.step
            info["steps/critic"] = self.critic.step
            info["anneal/ent_cf"] = self.ent_cf
          
        return self, info



    def update_critic(self, batch: Batch) -> tuple["BaselineSAC", dict]:
        b_nxt_control, b_nxt_logprob =jax.vmap(lambda obs, z: self.policy.apply(obs, z).experimental_sample_and_log_prob(seed=self.key))(batch.b_nxt_obs, batch.b_nxt_z)

        b_nxt_critic_all= jax.vmap(self.critic.apply)(batch.b_nxt_obs, batch.b_nxt_z)
        b_nxt_critics = jax.vmap(lambda nxt_critic, nxt_control: nxt_critic[:, nxt_control], in_axes = 0)(b_nxt_critic_all, b_nxt_control).reshape(-1, self.cfg.net.n_critics)
        
        b_nxt_critic = jnp.min(b_nxt_critics, axis = 1)

        # reward = - l
        b_target_critic = (- batch.b_l) + self.disc_gamma * b_nxt_critic
        b_target_critic -= self.disc_gamma * b_nxt_logprob 
        
        def get_critic_loss(params):
            b_critic_all = jax.vmap(ft.partial(self.critic.apply_with, params=params))(batch.b_obs, batch.b_z)
            b_critics = jax.vmap(lambda critic_1_2, control: critic_1_2[:, control], in_axes = 0)(b_critic_all, batch.b_control).reshape(-1, self.cfg.net.n_critics) 
            
            assert b_target_critic.shape[0] == b_critics.shape[0] 
 
            loss_critics = (b_critics - b_target_critic[:, jnp.newaxis]) ** 2

            info = {f"Loss/critic_{i}": loss_critics[:, i].mean() for i in range(self.cfg.net.n_critics)} | {f"mean_critic_{i}": b_critics[:, i].mean() for i in range(self.cfg.net.n_critics)}

            loss_critic = jnp.mean(loss_critics)
            info.update({'Loss/critic': loss_critic})

            return loss_critic, info

        grads_critic, critic_info = jax.grad(get_critic_loss, has_aux=True)(self.critic.params)
        grads_critic, critic_info["Grad/critic"] = compute_norm_and_clip(grads_critic, self.train_cfg.clip_grad_V)
 
        # Compute the kernel matrix and kernel gradients 
        ensemble_critic_params = self.critic.params['DoubleDiscreteCriticNet_0'] 
        ensemble_critic_grads = grads_critic['DoubleDiscreteCriticNet_0']
        kernel_matrix = compute_kernel_matrix(ensemble_critic_params)
        kernel_gradients = compute_kernel_gradient(ensemble_critic_params)

        # Compute SVGD updates
        ensemble_critic_grads = svgd_update(ensemble_critic_grads, kernel_matrix, kernel_gradients)
        grads_critic['DoubleDiscreteCriticNet_0'] = ensemble_critic_grads

        grads_critic, critic_info["Grad/critic"] = compute_norm_and_clip(grads_critic, self.train_cfg.clip_grad_V)
        critic = self.critic.apply_gradients(grads=grads_critic)
        
        new_target_critic_params = jax.tree_map(lambda p, tp: p * 5e-3 + tp * (1 - 5e-3), self.critic.params, self.target_critic.params)
        target_critic = self.target_critic.replace(params=new_target_critic_params)
        return self.replace(critic=critic, target_critic=target_critic), critic_info
     

    def update_policy(self, batch: Batch) -> tuple["BaselineSAC", dict]:
        def get_pol_loss(pol_params):
            pol_apply = ft.partial(self.policy.apply_with, params=pol_params)

            def get_logprob_entropy(obs, z, expert_control):
                dist = pol_apply(obs, z)
                expert_logprob = dist.log_prob(expert_control)

                entropy = dist.entropy()
                sampled_control, sampled_logprob = self.policy.apply(obs, z).experimental_sample_and_log_prob(seed=self.key) 

                return entropy, sampled_control, sampled_logprob, expert_logprob
            
            b_entropy, b_sampled_control, b_sampled_logprob, b_expert_logprob = jax.vmap(get_logprob_entropy)(batch.b_obs, batch.b_z, batch.b_expert_control)
            b_critic_all = jax.vmap(self.critic.apply)(batch.b_obs, batch.b_z)
            b_critics = jax.vmap(lambda critic, control: critic[:, control], in_axes = 0)(b_critic_all, b_sampled_control).reshape(-1, self.cfg.net.n_critics)
           
            b_critic = jnp.min(b_critics, axis = 1)

            sac_loss = jnp.mean(b_sampled_logprob * self.temp - b_critic) 

            bc_loss = - jnp.mean(b_expert_logprob)

            pol_loss =  (1 - bc_ratio) * sac_loss + bc_ratio * bc_loss
  
            mean_entropy = b_entropy.mean()
            
            info = {
                "loss_entropy": mean_entropy,
                "loss_sac": sac_loss,
                "loss_bc": bc_loss,
                "loss_pol": pol_loss
            }
            return pol_loss, info

        ent_cf = self.ent_cf
        bc_ratio = self.train_cfg.bc_ratio
        grads, pol_info = jax.grad(get_pol_loss, has_aux=True)(self.policy.params)
        
        grads, pol_info["Grad/pol"] = compute_norm_and_clip(grads, self.train_cfg.clip_grad_pol)
        policy = self.policy.apply_gradients(grads=grads)

        new_temp = self.temp - 1e-3 *  ent_cf * pol_info['loss_entropy'] * self.temp
        #pol_info["temperature"] = new_temp
        return self.replace(policy=policy, temp = new_temp), pol_info

    @ft.partial(jax.jit, donate_argnums=1)
    def collect(self, collector: Collector) -> tuple[Collector, RolloutOutput]:
        z_min, z_max = self.train_cfg.z_min, self.train_cfg.z_max 
        return collector.collect_batch(ft.partial(self.policy.apply), self.disc_gamma, z_min, z_max)

    def collect_iteratively(self, collector: Collector, replay_buffer: ReplayBuffer) -> tuple[Collector, Batch]:
        z_min, z_max = self.train_cfg.z_min, self.train_cfg.z_max
        collector, data = collector.collect_batch_iteratively(ft.partial(self.policy.apply), self.disc_gamma, z_min, z_max)
        # Compute GAE values.
        replay_buffer, batch = self.make_dset(replay_buffer, data)

        n_batches = self.train_cfg.n_batches
        assert n_batches == batch.num_batches
        batch_size = batch.batch_size  
        logger.info(f"Using {n_batches} x {batch_size} minibatches each epoch!")

        return collector, replay_buffer, batch

    def eval_iteratively(self, rollout_T: int) -> EvalData:
        # Evaluate for a range of zs.
        val_zs = np.linspace(self.train_cfg.z_min, self.train_cfg.z_max, num=8)

        Z_datas = []
        for z in val_zs:
            data = self.eval_single_z_iteratively(z, rollout_T)
            Z_datas.append(data)
        Z_data = tree_stack(Z_datas)

        info = jtu.tree_map(lambda arr: {"l2go=0": arr[0], "l2go=4": arr[4], "l2go=7": arr[7]}, Z_data.info)
        info["update_idx"] = self.update_idx
        return Z_data._replace(info=info)
    

    @ft.partial(jax.jit, static_argnames=["rollout_T"])
    def eval(self, rollout_T: int) -> EvalData:
        # Evaluate for a range of zs.
        val_zs = np.linspace(self.train_cfg.z_min, self.train_cfg.z_max, num=8)

        Z_datas = []
        for z in val_zs:
            data = self.eval_single_z(z, rollout_T)
            Z_datas.append(data)
        Z_data = tree_stack(Z_datas)

        info = jtu.tree_map(lambda arr: {"l2go=0": arr[0], "l2go=4": arr[4], "l2go=7": arr[7]}, Z_data.info)
        info["update_idx"] = self.update_idx
        return Z_data._replace(info=info)

    def get_mode_and_prob(self, obs, z):
        dist = self.policy.apply(obs, z)
        mode_sample = dist.mode()
        mode_prob = jnp.exp(dist.log_prob(mode_sample))
        return mode_sample, mode_prob

    def get_target_critic(self, obs, z):
        h_target_critic = self.target_critic.apply(obs, z)
        return h_target_critic.max()

    def eval_single_z(self, z: float, rollout_T: int):
        # --------------------------------------------
        # Plot value functions.
        bb_X, bb_Y, bb_state = self.task.grid_contour()
        bb_obs = jax_vmap(self.task.get_obs, rep=2)(bb_state)
        bb_z = jnp.full(bb_X.shape, z)

        bb_pol, bb_prob = jax_vmap(self.get_mode_and_prob, rep=2)(bb_obs, bb_z)
        bb_critic = jax_vmap(self.critic.apply, rep=2)(bb_obs, bb_z)
        bb_target_critic = jax_vmap(self.get_target_critic, rep=2)(bb_obs, bb_z)

        # --------------------------------------------
        # Rollout trajectories and get stats.
        z_min, z_max = self.train_cfg.z_min, self.train_cfg.z_max

        b_x0 = self.task.get_x0_eval()
        batch_size = len(b_x0)
        b_z0 = jnp.full(batch_size, z)

        collect_fn = ft.partial(
            collect_single_mode,
            self.task,
            get_pol=self.policy.apply,
            disc_gamma=self.disc_gamma,
            z_min=z_min,
            z_max=z_max,
            rollout_T=rollout_T,
        )
        b_rollout: RolloutOutput = jax_vmap(collect_fn)(b_x0, b_z0)

        b_h = jnp.max(b_rollout.Th_h, axis=(1, 2))
        assert b_h.shape == (batch_size,)
        b_issafe = b_h <= 0
        p_unsafe = 1 - b_issafe.mean()
        h_mean = jnp.mean(b_h)

        b_l = jnp.sum(b_rollout.T_l, axis=1)
        l_mean = jnp.mean(b_l)

        b_l_final = b_rollout.T_l[:, -1]
        l_final = jnp.mean(b_l_final)
        # --------------------------------------------

        info = {"p_unsafe": p_unsafe, "h_mean": h_mean, "cost sum": l_mean, "l_final": l_final}
        return self.EvalData(z, bb_pol, bb_prob, bb_critic, bb_target_critic, b_rollout.Tp1_state, info)

    def eval_single_z_iteratively(self, z: float, rollout_T: int):
        # --------------------------------------------
        # Plot value functions.
        bb_X, bb_Y, bb_state = jax2np(self.task.grid_contour())
        bb_obs = []
        for i in range(bb_state.shape[0]):
            bb_obs.append([])
            for j in range(bb_state.shape[1]): 
                bb_obs[-1].append(self.task.get_obs(bb_state[i][j]))
            bb_obs[-1] = jtu.tree_map(lambda *x: jnp.stack(x), *bb_obs[-1]) 
        bb_obs = jtu.tree_map(lambda *x: jnp.stack(x), *bb_obs)
        bb_z = jnp.full(bb_X.shape, z)

        bb_pol = []
        bb_prob = []  
        bb_critic = []  
        bb_target_critic = [] 

        for i in range(bb_obs.shape[0]):
            for bb in [bb_pol, bb_prob, bb_critic, bb_target_critic]:
                bb.append([])
                 
            for j in range(bb_obs.shape[1]):
                pol, prob = self.get_mode_and_prob(bb_obs[i][j], bb_z[i][j])
                bb_pol[-1].append(pol)
                bb_prob[-1].append(prob)
                
                critic_all = self.critic.apply(bb_obs[i][j], bb_z[i][j])
                critics = jax.vmap(lambda critic: critic.flatten()[pol], in_axes = 0)(critic_all).reshape(-1, self.cfg.net.n_critics)
                critic = jnp.min(critics, axis = 1).item()
                #logger.info(f"{i=}, {j=}, {critic=}")
                bb_critic[-1].append(critic)

                target_critic_all = self.target_critic.apply(bb_obs[i][j], bb_z[i][j])
                target_critics =jax.vmap(lambda target_critic: target_critic.flatten()[pol], in_axes = 0)(target_critic_all).reshape(-1, self.cfg.net.n_critics)
                target_critic = jnp.min(target_critics, axis = 1).item()
                #logger.info(f"{i=}, {j=}, {target_critic=}")
                bb_target_critic[-1].append(target_critic)

            for bb in [bb_pol, bb_prob, bb_critic, bb_target_critic]:
                bb[-1] = jtu.tree_map(lambda *x: jnp.stack(x), *bb[-1])
        
        bb_pol = jtu.tree_map(lambda *x: jnp.stack(x), *bb_pol)
        bb_prob = jtu.tree_map(lambda *x: jnp.stack(x), *bb_prob)
        bb_critic = jtu.tree_map(lambda *x: jnp.stack(x), *bb_critic)
        bb_target_critic = jtu.tree_map(lambda *x: jnp.stack(x), *bb_target_critic)
         
        # --------------------------------------------
        # Rollout trajectories and get stats.
        z_min, z_max = self.train_cfg.z_min, self.train_cfg.z_max

        b_x0 = self.task.get_x0_eval()
        batch_size = len(b_x0)
        b_z0 = jnp.full(batch_size, z)

        collect_fn = ft.partial(
            collect_single_env_mode,
            self.task,
            get_pol=self.policy.apply,
            disc_gamma=self.disc_gamma,
            z_min=z_min,
            z_max=z_max,
            rollout_T=rollout_T,
        )
        bb_rollouts: list[RolloutOutput] = []
        for i in range(batch_size):
            bb_rollouts.append(collect_fn(b_x0[i], b_z0[i]))
        b_rollout: RolloutOutput = jtu.tree_map(lambda *x: jnp.stack(x), *bb_rollouts)
 
        b_h = jnp.max(b_rollout.Th_h, axis=(1, 2))
        assert b_h.shape == (batch_size,)
        b_issafe = b_h <= 0
        p_unsafe = 1 - b_issafe.mean()
        h_mean = jnp.mean(b_h)

        b_l = jnp.sum(b_rollout.T_l, axis=1)
        l_mean = jnp.mean(b_l)

        b_l_final = b_rollout.T_l[:, -1]
        l_final = jnp.mean(b_l_final)
        # --------------------------------------------

        info = {"p_unsafe": p_unsafe, "h_mean": h_mean, "cost sum": l_mean, "l_final": l_final}
        return self.EvalData(z, bb_pol, bb_prob, bb_critic, bb_target_critic, b_rollout.Tp1_state, info)

 