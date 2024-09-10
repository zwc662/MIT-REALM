import functools as ft
from typing import NamedTuple, TypeVar, Generic, Optional
from dataclasses import dataclass

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

from efppo.networks.ef_wrapper import EFWrapper, ZEncoder, BoltzmanPolicyWrapper, EnsembleBoltzmanPolicyWrapper
from efppo.networks.mlp import MLP
from efppo.networks.network_utils import ActLiteral, HidSizes, get_act_from_str, get_default_tx, rsample
from efppo.networks.policy_net import DiscretePolicyNet, EnsembleDiscretePolicyNet
from efppo.networks.train_state import TrainState 
from efppo.networks.critic_net import DiscreteCriticNet, EnsembleDiscreteCriticNet
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


_Algo = TypeVar("Baseline_Algorithm")


@define
class BaselineCfg(Cfg):
    @define
    class TrainCfg(Cfg):
        z_min: float
        z_max: float

        n_batches: int
        batch_size: int
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
       
    alg: str
    net: NetCfg
    train: TrainCfg
    eval: EvalCfg


class Baseline(Generic[_Algo], struct.PyTreeNode):
    update_idx: int
    key: PRNGKey 
    temp: FloatScalar
    policy: TrainState[tfd.Distribution]

    disc_gamma: FloatScalar

    #task: Task = struct.field(pytree_node=False)
    cfg: BaselineCfg = struct.field(pytree_node=False)
    target_ent: float = struct.field(pytree_node=False)
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
        
        zbT_x: ZBTState

        info: dict[str, float]

    @classmethod
    def create(cls, key: jr.PRNGKey, task: Task, cfg: BaselineCfg):
        key, key_pol = jr.split(key, 2)

        obs, z = np.zeros(task.nobs), np.array(0.0)
        act = get_act_from_str(cfg.net.act)

        # Encoder for z. Params not shared.
        z_base_cls = ft.partial(ZEncoder, nz=cfg.net.nz_enc, z_mean=cfg.net.z_mean, z_scale=cfg.net.z_scale)
        
        # Define policy network.
        pol_base_cls = ft.partial(MLP, cfg.net.pol_hids, act, act_final=cfg.net.act_final, scale_final=1e-2)
        pol_cls = ft.partial(DiscretePolicyNet, pol_base_cls, task.n_actions)
        pol_def = EFWrapper(pol_cls, z_base_cls)
        pol_tx = get_default_tx(as_schedule(cfg.net.pol_lr).make())
        pol = TrainState.create_from_def(key_pol, pol_def, (obs, z), pol_tx)
 
        ent_cf = as_schedule(cfg.net.entropy_cf).make()
        disc_gamma_sched = as_schedule(cfg.net.disc_gamma).make()
        disc_gamma = disc_gamma_sched(0)
        target_ent =  -task.n_actions / (task.n_actions + 1) * np.log(1/(task.n_actions+1))
 
        return Baseline(0, key, 1, pol, disc_gamma, cfg, target_ent, ent_cf, disc_gamma_sched)

       
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
    

    @ft.partial(jax.jit, donate_argnums=1)
    def collect(self, collector: Collector) -> tuple[Collector, RolloutOutput]:
        z_min, z_max = self.train_cfg.z_min, self.train_cfg.z_max 
        return collector.collect_batch(ft.partial(self.policy.apply), self.disc_gamma, z_min, z_max)

    def collect_iteratively(self, collector: Collector, rollout_T: Optional[int] = None) -> tuple[Collector, Batch]:
        z_min, z_max = self.train_cfg.z_min, self.train_cfg.z_max
        collector, data = collector.collect_batch_iteratively(ft.partial(self.policy.apply), self.disc_gamma, z_min, z_max, rollout_T)
        return collector, data
    
    def make_dset(self, replay_buffer: ReplayBuffer) -> Batch:
        num_batches = self.train_cfg.n_batches
        batch_size =  self.train_cfg.batch_size
        batch_data = replay_buffer.sample(num_batches, batch_size)
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
        return b_batch

    def eval_iteratively(self, task: Task, rollout_T: int) -> EvalData:
        # Evaluate for a range of zs.
        val_zs = np.linspace(self.train_cfg.z_min, self.train_cfg.z_max, num=8)

        Z_datas = []
        for z in val_zs:
            data = self.eval_single_z_iteratively(task, z, rollout_T)
            Z_datas.append(data)
        Z_data = tree_stack(Z_datas)

        info = jtu.tree_map(lambda arr: {"l2go=0": arr[0], "l2go=4": arr[4], "l2go=7": arr[7]}, Z_data.info)
        info["update_idx"] = self.update_idx
        return Z_data._replace(info=info)
    
    @ft.partial(jax.jit, static_argnames=["rollout_T"])
    def eval(self, task: Task, rollout_T: int) -> EvalData:
        # Evaluate for a range of zs.
        val_zs = np.linspace(self.train_cfg.z_min, self.train_cfg.z_max, num=8)

        Z_datas = []
        for z in val_zs:
            data = self.eval_single_z(task, z, rollout_T)
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

    
    def eval_single_z(self, task: Task, z: float, rollout_T: int):
        # --------------------------------------------
        # Plot value functions.
        bb_X, bb_Y, bb_state = task.grid_contour()
        bb_obs = jax_vmap(task.get_obs, rep=2)(bb_state)
        bb_z = jnp.full(bb_X.shape, z)

        bb_pol, bb_prob = jax_vmap(self.get_mode_and_prob, rep=2)(bb_obs, bb_z) 

        # --------------------------------------------
        # Rollout trajectories and get stats.
        z_min, z_max = self.train_cfg.z_min, self.train_cfg.z_max

        b_x0 = task.get_x0_eval()
        batch_size = len(b_x0)
        b_z0 = jnp.full(batch_size, z)

        collect_fn = ft.partial(
            collect_single_mode,
            task,
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
        return self.EvalData(z, bb_pol, bb_prob, b_rollout.Tp1_state, info)
     
    def eval_single_z_iteratively(self, task: Task, z: float, rollout_T: int):
        # --------------------------------------------
        # Plot value functions.
        bb_X, bb_Y, bb_state = jax2np(task.grid_contour())
        bb_obs = []
        for i in range(bb_state.shape[0]):
            bb_obs.append([])
            for j in range(bb_state.shape[1]): 
                bb_obs[-1].append(task.get_obs(bb_state[i][j]))
            bb_obs[-1] = jtu.tree_map(lambda *x: jnp.stack(x), *bb_obs[-1]) 
        bb_obs = jtu.tree_map(lambda *x: jnp.stack(x), *bb_obs)
        bb_z = jnp.full(bb_X.shape, z)

        bb_pol = []
        bb_prob = []   

        for i in range(bb_obs.shape[0]):
            for bb in [bb_pol, bb_prob]:
                bb.append([])
                 
            for j in range(bb_obs.shape[1]):
                pol, prob = self.get_mode_and_prob(bb_obs[i][j], bb_z[i][j])
                bb_pol[-1].append(pol)
                bb_prob[-1].append(prob)
                
            for bb in [bb_pol, bb_prob]:
                bb[-1] = jtu.tree_map(lambda *x: jnp.stack(x), *bb[-1])
        
        bb_pol = jtu.tree_map(lambda *x: jnp.stack(x), *bb_pol)
        bb_prob = jtu.tree_map(lambda *x: jnp.stack(x), *bb_prob) 
        # --------------------------------------------
        # Rollout trajectories and get stats.
        z_min, z_max = self.train_cfg.z_min, self.train_cfg.z_max

        b_x0 = task.get_x0_eval()
        batch_size = len(b_x0)
        b_z0 = jnp.full(batch_size, z)

        collect_fn = ft.partial(
            collect_single_env_mode,
            task,
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
        return self.EvalData(z, bb_pol, bb_prob, b_rollout.Tp1_state, info)


class BaselineSAC(Baseline):
    critic: TrainState[LFloat]
    target_critic: TrainState[HFloat]

    class EvalData(NamedTuple):
        z_zs: ZFloat
        zbb_pol: ZBBControl
        zbb_prob: ZBBFloat
        
        zbT_x: ZBTState

        info: dict[str, float]
        
        zbb_critic: ZBBFloat
        zbb_target_critic: ZBBFloat
        
     
    @classmethod
    def create(cls, key: jr.PRNGKey, task: Task, cfg: BaselineCfg):
        key, key_critic = jr.split(key, 2)

        baseline = super(BaselineSAC, cls).create(key, task, cfg)
        base_kwargs = {k: getattr(baseline, k) for k in baseline.__dataclass_fields__}


        obs, z = np.zeros(task.nobs), np.array(0.0)
        act = get_act_from_str(cfg.net.act)

        # Encoder for z. Params not shared.
        z_base_cls = ft.partial(ZEncoder, nz=cfg.net.nz_enc, z_mean=cfg.net.z_mean, z_scale=cfg.net.z_scale)
        
        # Define critic network.
        print(f"Ensembled {cfg.net.n_critics} critic networks")
        critic_base_cls = ft.partial(MLP, cfg.net.val_hids, act)
        critic_cls = ft.partial(EnsembleDiscreteCriticNet, critic_base_cls, task.n_actions, cfg.net.n_critics)
        critic_def = EFWrapper(critic_cls, z_base_cls)
        critic_tx = get_default_tx(as_schedule(cfg.net.val_lr).make())
        critic = TrainState.create_from_def(key_critic, critic_def, (obs, z), critic_tx)

        # Define target_critic network.
        target_critic_base_cls = ft.partial(MLP, cfg.net.val_hids, act)
        target_critic_cls = ft.partial(EnsembleDiscreteCriticNet, target_critic_base_cls, task.n_actions, cfg.net.n_critics)
        target_critic_def = EFWrapper(target_critic_cls, z_base_cls)
        target_critic_tx = get_default_tx(as_schedule(cfg.net.val_lr).make())
        target_critic = TrainState.create_from_def(key_critic,  target_critic_def, (obs, z), target_critic_tx)

        return BaselineSAC(critic = critic, target_critic = target_critic, **base_kwargs)

    

    @ft.partial(jax.jit, donate_argnums=0)
    def update(self, replay_buffer: ReplayBuffer) -> tuple["BaselineSAC", dict]:
        batch = self.make_dset(replay_buffer)
 
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
    

    def update_iteratively(self, replay_buffer: ReplayBuffer) -> tuple["BaselineSAC", dict]:
        batch = self.make_dset(replay_buffer)
 
        info = {}
        for batch_idx in range(batch.num_batches):
            key_shuffle, key_self = jr.split(self.key, 2)
            self = self.replace(key = key_self)
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



    def update_critic(self, batch: Baseline.Batch) -> tuple["BaselineSAC", dict]:
        b_nxt_control, b_nxt_logprob =jax.vmap(lambda obs, z: self.policy.apply(obs, z).experimental_sample_and_log_prob(seed=self.key))(batch.b_nxt_obs, batch.b_nxt_z)

        b_nxt_critic_all= jax.vmap(self.critic.apply)(batch.b_nxt_obs, batch.b_nxt_z)
        b_nxt_critics = jax.vmap(lambda nxt_critic, nxt_control: nxt_critic[:, nxt_control], in_axes = 0)(b_nxt_critic_all, b_nxt_control).reshape(-1, self.cfg.net.n_critics)
        
        b_nxt_critic = jnp.max(b_nxt_critics, axis = 1)
 
        b_target_critic = batch.b_l + self.disc_gamma * b_nxt_critic
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
        ensemble_critic_params = self.critic.params['EnsembleDiscreteCriticNet_0'] 
        ensemble_critic_grads = grads_critic['EnsembleDiscreteCriticNet_0']
        kernel_matrix = compute_kernel_matrix(ensemble_critic_params)
        kernel_gradients = compute_kernel_gradient(ensemble_critic_params)

        # Compute SVGD updates
        ensemble_critic_grads = svgd_update(ensemble_critic_grads, kernel_matrix, kernel_gradients)
        grads_critic['EnsembleDiscreteCriticNet_0'] = ensemble_critic_grads

        grads_critic, critic_info["Grad/critic"] = compute_norm_and_clip(grads_critic, self.train_cfg.clip_grad_V)
        critic = self.critic.apply_gradients(grads=grads_critic)
        
        new_target_critic_params = jax.tree_map(lambda p, tp: p * 5e-3 + tp * (1 - 5e-3), self.critic.params, self.target_critic.params)
        target_critic = self.target_critic.replace(params=new_target_critic_params)
        return self.replace(critic=critic, target_critic=target_critic), critic_info
     

    def update_policy(self, batch: Baseline.Batch) -> tuple["BaselineSAC", dict]:
        def get_pol_loss(pol_params):
            pol_apply = ft.partial(self.policy.apply_with, params=pol_params)

            def get_logprob_entropy(obs, z, control):
                dist = pol_apply(obs, z)
                logprob = dist.log_prob(control)
                entropy = dist.entropy()
                logits = dist.logits
                return entropy, logits, logprob
            
            b_entropy, b_logprobs, b_expert_logprob = jax.vmap(get_logprob_entropy)(batch.b_obs, batch.b_z, batch.b_expert_control)
            b_critic_all = jax.vmap(self.critic.apply)(batch.b_obs, batch.b_z)
            b_critics = jax.vmap(
                lambda critics_all, logprobs: jax.vmap(lambda critics: critics @ jnp.exp(logprobs), in_axes = 0)(critics_all), in_axes = 0)(
                    b_critic_all, b_logprobs).reshape(-1, self.cfg.net.n_critics)
           
            b_critic = jnp.max(b_critics, axis = 1)

            sac_loss = jnp.mean(- b_entropy * self.temp + b_critic) 

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

        new_temp = self.temp - ent_cf * (pol_info['loss_entropy'] - self.target_ent)
        pol_info["temperature"] = new_temp
        return self.replace(policy=policy, temp = new_temp), pol_info


    def get_target_critic(self, obs, z):
        h_target_critic = self.target_critic.apply(obs, z)
        return h_target_critic.max()

    def eval_single_z(self, task: Task, z: float, rollout_T: int):
        # --------------------------------------------
        # Plot value functions.
        bb_X, bb_Y, bb_state = task.grid_contour()
        bb_obs = jax_vmap(task.get_obs, rep=2)(bb_state)
        bb_z = jnp.full(bb_X.shape, z)

        bb_pol, bb_prob = jax_vmap(self.get_mode_and_prob, rep=2)(bb_obs, bb_z)
        bb_critic = jax_vmap(self.critic.apply, rep=2)(bb_obs, bb_z)
        bb_target_critic = jax_vmap(self.get_target_critic, rep=2)(bb_obs, bb_z)

        # --------------------------------------------
        # Rollout trajectories and get stats.
        z_min, z_max = self.train_cfg.z_min, self.train_cfg.z_max

        b_x0 = task.get_x0_eval()
        batch_size = len(b_x0)
        b_z0 = jnp.full(batch_size, z)

        collect_fn = ft.partial(
            collect_single_mode,
            task,
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
        return self.EvalData(z, bb_pol, bb_prob, b_rollout.Tp1_state, info, zbb_critic = bb_critic, zbb_target_critic = bb_target_critic)
     
    def eval_single_z_iteratively(self, task: Task, z: float, rollout_T: int):
        # --------------------------------------------
        # Plot value functions.
        bb_X, bb_Y, bb_state = jax2np(task.grid_contour())
        bb_obs = []
        for i in range(bb_state.shape[0]):
            bb_obs.append([])
            for j in range(bb_state.shape[1]): 
                bb_obs[-1].append(task.get_obs(bb_state[i][j]))
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

        b_x0 = task.get_x0_eval()
        batch_size = len(b_x0)
        b_z0 = jnp.full(batch_size, z)

        collect_fn = ft.partial(
            collect_single_env_mode,
            task,
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
        return BaselineSAC.EvalData(z, bb_pol, bb_prob, b_rollout.Tp1_state, info, zbb_critic = bb_critic, zbb_target_critic = bb_target_critic)
     



class BaselineGumbelSAC(BaselineSAC):
    critic_std: TrainState[LFloat]
     
    @classmethod
    def create(cls, key: jr.PRNGKey, task: Task, cfg: BaselineCfg):
        key_critic_std, key = jr.split(key, 2)
        
        baseline = super(BaselineGumbelSAC, cls).create(key, task, cfg)
        base_kwargs = {k: getattr(baseline, k) for k in baseline.__dataclass_fields__}
       
        obs, z = np.zeros(task.nobs), np.array(0.0)
        act = get_act_from_str(cfg.net.act)

        # Encoder for z. Params not shared.
        z_base_cls = ft.partial(ZEncoder, nz=cfg.net.nz_enc, z_mean=cfg.net.z_mean, z_scale=cfg.net.z_scale)
       
        # Define critic network.
        print(f"Ensembled {cfg.net.n_critics} critic networks")
        critic_base_cls = ft.partial(MLP, cfg.net.val_hids, act)
        critic_cls = ft.partial(EnsembleDiscreteCriticNet, critic_base_cls, task.n_actions, cfg.net.n_critics)
        critic_def = EFWrapper(critic_cls, z_base_cls)
        critic_tx = get_default_tx(as_schedule(cfg.net.val_lr).make())
        critic_std = TrainState.create_from_def(key_critic_std, critic_def, (obs, z), critic_tx)

        return BaselineGumbelSAC(critic_std = critic_std, **base_kwargs) 
    
    @classmethod
    def create(cls, key: jr.PRNGKey, task: Task, cfg: BaselineCfg):
        key, key_pol, key_critic = jr.split(key, 3)

        obs, z = np.zeros(task.nobs), np.array(0.0)
        act = get_act_from_str(cfg.net.act)

        # Encoder for z. Params not shared.
        z_base_cls = ft.partial(ZEncoder, nz=cfg.net.nz_enc, z_mean=cfg.net.z_mean, z_scale=cfg.net.z_scale)
       
        baseline = super(BaselineSAC, cls).create(key, task, cfg)
        base_kwargs = {k: getattr(baseline, k) for k in baseline.__dataclass_fields__}
       
 
        # Define critic network.
        print(f"Ensembled {cfg.net.n_critics} critic networks")
        critic_base_cls = ft.partial(MLP, cfg.net.val_hids, act)
        critic_cls = ft.partial(EnsembleDiscreteCriticNet, critic_base_cls, task.n_actions, cfg.net.n_critics)
        critic_def = EFWrapper(critic_cls, z_base_cls)
        critic_tx = get_default_tx(as_schedule(cfg.net.val_lr).make())
        critic = TrainState.create_from_def(key_critic, critic_def, (obs, z), critic_tx)

        # Define target_critic network.
        target_critic_base_cls = ft.partial(MLP, cfg.net.val_hids, act)
        target_critic_cls = ft.partial(EnsembleDiscreteCriticNet, target_critic_base_cls, task.n_actions, cfg.net.n_critics)
        target_critic_def = EFWrapper(target_critic_cls, z_base_cls)
        target_critic_tx = get_default_tx(as_schedule(cfg.net.val_lr).make())
        target_critic = TrainState.create_from_def(key_critic,  target_critic_def, (obs, z), target_critic_tx)

        return BaselineSAC(critic = critic, target_critic = target_critic, **base_kwargs)


    def update_critic(self, batch: Baseline.Batch) -> tuple["BaselineGumbelSAC", dict]:
        raise NotImplementedError


class BaselineDQN(Baseline):
    critic: TrainState[LFloat]
    target_critic: TrainState[LFloat]
    
    class EvalData(NamedTuple):
        z_zs: ZFloat
        zbb_pol: ZBBControl
        zbb_prob: ZBBFloat
        
        zbT_x: ZBTState

        info: dict[str, float]
        zbb_critic: ZBBFloat
     
    @classmethod
    def create(cls, key: jr.PRNGKey, task: Task, cfg: BaselineCfg):
        key, key_critic = jr.split(key, 2)

        obs, z = np.zeros(task.nobs), np.array(0.0)
        act = get_act_from_str(cfg.net.act)

        # Encoder for z. Params not shared.
        z_base_cls = ft.partial(ZEncoder, nz=cfg.net.nz_enc, z_mean=cfg.net.z_mean, z_scale=cfg.net.z_scale)
        
        ent_cf = as_schedule(cfg.net.entropy_cf).make()
        disc_gamma_sched = as_schedule(cfg.net.disc_gamma).make()
        disc_gamma = disc_gamma_sched(0)

        obs, z = np.zeros(task.nobs), np.array(0.0)
        act = get_act_from_str(cfg.net.act)
  
        # Define critic network.
        print(f"Ensembled {cfg.net.n_critics} critic networks")
        critic_base_cls = ft.partial(MLP, cfg.net.val_hids, act)
        critic_cls = ft.partial(EnsembleDiscreteCriticNet, critic_base_cls, task.n_actions, cfg.net.n_critics)
        critic_def = EFWrapper(critic_cls, z_base_cls)
        critic_tx = get_default_tx(as_schedule(cfg.net.val_lr).make())
        critic = TrainState.create_from_def(key_critic, critic_def, (obs, z), critic_tx)

        # Define target_critic network.
        target_critic_base_cls = ft.partial(MLP, cfg.net.val_hids, act)
        target_critic_cls = ft.partial(EnsembleDiscreteCriticNet, target_critic_base_cls, task.n_actions, cfg.net.n_critics)
        target_critic_def = EFWrapper(target_critic_cls, z_base_cls)
        target_critic_tx = get_default_tx(as_schedule(cfg.net.val_lr).make())
        target_critic = TrainState.create_from_def(key_critic,  target_critic_def, (obs, z), target_critic_tx)

        get_pol = EnsembleBoltzmanPolicyWrapper(critic_def)
        policy_tx = get_default_tx(as_schedule(cfg.net.pol_lr).make())
        baseline = super(BaselineDQN, cls).create(key, task, cfg)
        baseline = baseline.replace(policy = TrainState.create(get_pol, critic.params, policy_tx))
        base_kwargs = {k: getattr(baseline, k) for k in baseline.__dataclass_fields__}
 
        return BaselineDQN(critic = critic, target_critic = target_critic, **base_kwargs)

    @ft.partial(jax.jit, donate_argnums=0)
    def update(self, batch: Baseline.Batch) -> tuple["BaselineDQN", dict]:
        def update_one_batch(obj, batch_idx):
            key_shuffle, key_self = jr.split(obj.key, 2)
            rand_idxs = jr.permutation(key_shuffle, jnp.arange(batch.batch_size))
            mb_dset = jax.tree_map(lambda x: x[batch_idx][rand_idxs], batch)
            
            # 3: Perform value function and policy updates.
            def updates_body(alg_: BaselineDQN, b_batch: BaselineDQN.Batch):
                alg_, pol_info = alg_.update_critic(b_batch)
                return alg_, pol_info

            new_obj, info = lax.scan(updates_body, obj, mb_dset, length=batch.num_batches)
            # Take the mean.
            info = jax.tree_map(jnp.mean, info)

            info["steps/policy"] = obj.policy.step
            info["steps/critic"] = obj.critic.step
            info["anneal/ent_cf"] = obj.ent_cf

            return new_obj.replace(key=key_self, update_idx=obj.update_idx + 1), info
        
        new_self, new_info = lax.scan(update_one_batch, self, jnp.arange(batch.num_batches))

        new_self = new_self.update_target_critic(batch)
                
 
        return new_self, jtu.tree_map(jnp.mean, new_info)
    

    def update_iteratively(self, replay_buffer: ReplayBuffer) -> tuple["BaselineDQN", dict]:
        batch = self.make_dset(replay_buffer)
        info = None
        for batch_idx in range(batch.num_batches):
            key_shuffle, key_self = jr.split(self.key, 2)
            self = self.replace(key = key_self)
            rand_idxs = jr.permutation(key_shuffle, jnp.arange(batch.batch_size))
            mb_dset = jax.tree_map(lambda x: x[batch_idx][rand_idxs], batch)
            
            # 3: Perform value function and policy updates.
            def updates_body(alg_: BaselineDQN, b_batch: BaselineDQN.Batch):
                alg_, policy_info = alg_.update_policy(b_batch)  
                alg_, critic_info = alg_.update_critic(b_batch)

                return alg_,  critic_info | policy_info

            self, new_info = updates_body(self, mb_dset)
            # Take the mean.
            if info is None:
                info = jax.tree_map(lambda x: jnp.asarray([x]), new_info)
            else:
                info = jax.tree_map(lambda xs, x: jnp.stack((*xs, x)), info, new_info)

        info = jax.tree_map(jnp.mean, info)
        info["steps/critic"] = self.critic.step        
        
        self = self.update_target_critic()   
        
          
        return self, info
    
    def update_policy(self, batch: Baseline.Batch):
        new_policy_params = self.critic.params
        new_policy = self.policy.replace(params=new_policy_params)

        def get_pol_loss(pol_params):
            pol_apply = ft.partial(self.policy.apply_with, params=pol_params)

            def get_logprob_entropy(obs, z, expert_control):
                dist = pol_apply(obs, z)
                expert_logprob = dist.log_prob(expert_control)
                entropy = dist.entropy()
             
                return entropy, expert_logprob
            
            b_entropy,  b_expert_logprob = jax.vmap(get_logprob_entropy)(batch.b_obs, batch.b_z, batch.b_expert_control)
              
            bc_loss = - jnp.mean(b_expert_logprob)

            mean_entropy = b_entropy.mean()


            pol_loss = bc_ratio * bc_loss - ent_cf * mean_entropy
  
            
            info = {
                "loss_entropy": mean_entropy, 
                "loss_bc": bc_loss,
                "loss_pol": pol_loss
            }
            return pol_loss, info
        
        ent_cf = self.ent_cf
        bc_ratio = self.train_cfg.bc_ratio
        grads, pol_info = jax.grad(get_pol_loss, has_aux=True)(self.policy.params)
        
        grads, pol_info["Grad/pol"] = compute_norm_and_clip(grads, self.train_cfg.clip_grad_pol)
        new_policy = self.policy.apply_gradients(grads=grads)

        new_critic_params = new_policy.params
        new_critic= self.critic.replace(params=new_critic_params)

        return self.replace(policy=new_policy, critic=new_critic), pol_info

    def update_target_critic(self):
        new_target_critic_params = jax.tree_map(lambda p, tp: p * 5e-3 + tp * (1 - 5e-3), self.critic.params, self.target_critic.params)
        target_critic = self.target_critic.replace(params=new_target_critic_params)
        return self.replace(target_critic = target_critic)

    def update_critic(self, batch: Baseline.Batch) -> tuple["BaselineDQN", dict]:
        b_nxt_critic_all= jax.vmap(lambda obs, z: self.target_critic.apply(obs, z))(batch.b_nxt_obs, batch.b_nxt_z)
        b_nxt_critics = jax.vmap(lambda nxt_critic: jnp.min(nxt_critic, axis = -1), in_axes = 0)(b_nxt_critic_all).reshape(-1, self.cfg.net.n_critics)
        b_nxt_critic = jnp.max(b_nxt_critics, axis = 1)
 
        b_target_critic = batch.b_l + self.disc_gamma * b_nxt_critic
         
        def get_critic_loss(params):
            b_critic_all = jax.vmap(lambda obs, z: self.critic.apply_with(obs, z, params=params), in_axes = 0)(batch.b_obs, batch.b_z)
            b_critics = jax.vmap(lambda critic_all, control: critic_all[:, control], in_axes = 0)(b_critic_all, batch.b_control).reshape(-1, self.cfg.net.n_critics) 
            
            assert b_target_critic.shape[0] == b_critics.shape[0] 
 
            loss_critics = (b_critics - b_target_critic[:, jnp.newaxis]) ** 2

            info = {f"Loss/critic_{i}": loss_critics[:, i].mean() for i in range(self.cfg.net.n_critics)} | {f"mean_critic_{i}": b_critics[:,  i].mean() for i in range(self.cfg.net.n_critics)}

            loss_critic = jnp.mean(loss_critics)
            info.update({'Loss/critic': loss_critic})

            return loss_critic, info

        grads, info = jax.grad(get_critic_loss, has_aux=True)(self.critic.params)
        grads,  _ = compute_norm_and_clip(grads, self.train_cfg.clip_grad_V)
 
        # Compute the kernel matrix and kernel gradients 
        ensemble_params = self.critic.params['EnsembleDiscreteCriticNet_0'] 
        ensemble_grads = grads['EnsembleDiscreteCriticNet_0']
        kernel_matrix = compute_kernel_matrix(ensemble_params)
        kernel_gradients = compute_kernel_gradient(ensemble_params)

        # Compute SVGD updates
        ensemble_grads = svgd_update(ensemble_grads, kernel_matrix, kernel_gradients)
        grads['EnsembleDiscreteCriticNet_0'] = ensemble_grads

        grads, info["Grad/critic"] = compute_norm_and_clip(grads, self.train_cfg.clip_grad_V)
        critic = self.critic.apply_gradients(grads=grads)
        
        return self.replace(critic=critic), info
      
    
    def get_critic(self, obs, z):
        return self.critic.apply(obs, z)
        h_critic = self.critic.apply(obs, z) 
        return h_critic.max()

    def get_target_critic(self, obs, z):
        return self.target_critic.apply(obs, z) 
        h_target_critic = self.target_critic.apply(obs, z) 
        return h_target_critic.max()

    def eval_single_z(self, task: Task, z: float, rollout_T: int):
        # --------------------------------------------
        # Plot value functions.
        bb_X, bb_Y, bb_state = task.grid_contour()
        bb_obs = jax_vmap(task.get_obs, rep=2)(bb_state)
        bb_z = jnp.full(bb_X.shape, z)

        bb_pol, bb_prob = jax_vmap(self.get_mode_and_prob, rep=2)(bb_obs, bb_z)
        bb_critic = jax_vmap(self.get_critic, rep=2)(bb_obs, bb_z)
         
        # --------------------------------------------
        # Rollout trajectories and get stats.
        z_min, z_max = self.train_cfg.z_min, self.train_cfg.z_max

        b_x0 = task.get_x0_eval()
        batch_size = len(b_x0)
        b_z0 = jnp.full(batch_size, z)

        collect_fn = ft.partial(
            collect_single_mode,
            task,
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
        return BaselineDQN.EvalData(z, bb_pol, bb_prob, b_rollout.Tp1_state, info, zbb_critic = bb_critic)
     
    def eval_single_z_iteratively(self, task: Task, z: float, rollout_T: int):
        # --------------------------------------------
        # Plot value functions.
        bb_X, bb_Y, bb_state = jax2np(task.grid_contour())
        bb_obs = []
        for i in range(bb_state.shape[0]):
            bb_obs.append([])
            for j in range(bb_state.shape[1]): 
                bb_obs[-1].append(task.get_obs(bb_state[i][j]))
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
                
                critic_all = self.get_critic(bb_obs[i][j], bb_z[i][j])
                critics = jax.vmap(lambda critic: critic.flatten()[pol], in_axes = 0)(critic_all).reshape(-1, self.cfg.net.n_critics)
                critic = jnp.min(critics, axis = 1).item()
                #logger.info(f"{i=}, {j=}, {critic=}")
                bb_critic[-1].append(critic) 

            for bb in [bb_pol, bb_prob, bb_critic, bb_target_critic]:
                bb[-1] = jnp.asarray(bb[-1])
        
        bb_pol = jnp.asarray(bb_pol)
        bb_prob = jnp.asarray(bb_prob) #jtu.tree_map(lambda x: jnp.stack(x), *bb_prob)
        bb_critic = jnp.asarray(bb_critic) #jtu.tree_map(lambda x: jnp.stack(x), *bb_critic)
        bb_target_critic = jnp.asarray(bb_target_critic) #jtu.tree_map(lambda x: jnp.stack(x), *bb_target_critic)
         
        # --------------------------------------------
        # Rollout trajectories and get stats.
        z_min, z_max = self.train_cfg.z_min, self.train_cfg.z_max

        b_x0 = task.get_x0_eval()
        batch_size = len(b_x0)
        b_z0 = jnp.full(batch_size, z)

        collect_fn = ft.partial(
            collect_single_env_mode,
            task,
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
        return BaselineDQN.EvalData(z, bb_pol, bb_prob, b_rollout.Tp1_state, info = info, zbb_critic = bb_critic)


    def eval_iteratively(self, task: Task, rollout_T: int) -> EvalData:
        # Evaluate for a range of zs.
        val_zs = np.linspace(self.train_cfg.z_min, self.train_cfg.z_max, num=8)

        Z_datas = []
        for z in val_zs:
            data = self.eval_single_z_iteratively(task, z, rollout_T)
            Z_datas.append(data)
        Z_data = tree_stack(Z_datas)
        #Z_data = jtu.tree_map(lambda *arr: jnp.stack(arr), *Z_datas)
        
        info = jtu.tree_map(lambda arr: {f"l2go={val_zs[0]}": arr[0], f"l2go={val_zs[4]}": arr[4], f"l2go={val_zs[7]}": arr[7]}, Z_data.info)
        info["update_idx"] = self.update_idx
        return Z_data._replace(info=info) 