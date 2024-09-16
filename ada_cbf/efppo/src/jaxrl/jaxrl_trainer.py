import os
import random
import time

import numpy as np
import tqdm 

from typing import Optional, Union
from efppo.utils.ckpt_utils import get_ckpt_manager_sync
from efppo.utils.jax_utils import jax2np, move_tree_to_cpu
from efppo.utils.path_utils import get_runs_dir, mkdir
from efppo.task.f110 import F1TenthWayPoint

  
from jaxrl.agents import (AWACLearner, DDPGLearner, REDQLearner, SACLearner,
                          SACV1Learner)
from jaxrl.datasets import ReplayBuffer
from jaxrl.evaluation import evaluate 
import jax.random as jr

from efppo.utils.rng import PRNGKey

from dataclasses import dataclass, asdict

import wandb
 




@dataclass
class JAXRLTrainer:
    name: str = 'sac'
    seed: int = 42
    eval_episodes: int  = 10
    log_interval: int = int(1e3)        # ('Logging interval.')
    eval_interval: int = int(5e3)       # ('Eval interval.')
    batch_size: int = 256        # ('Mini batch size.')
    updates_per_step: int = 1        # ('Gradient updates per step.')
    max_steps: int = int(1e7)        # ('Number of training steps.')
    start_training: int = int(1e4)      #'Number of training steps to start training.')
    tqdm: bool = True
    replay_buffer_size: int = int(1e6)
    agent: Optional[Union[AWACLearner, DDPGLearner, REDQLearner, SACLearner, SACV1Learner]] = None

    def train(self):
        kwargs = asdict(self)
        name = kwargs.pop('name')
        if 'awac' in name.lower():
            algo = AWACLearner
        elif 'ddpg' in name.lower():
            algo = DDPGLearner
        elif 'ql' in name.lower():
            algo = REDQLearner
        else:
            algo = SACV1Learner
 
        wandb.init(
            project='jaxrl',
            entity='zwc662',
            name=name,
            monitor_gym=True,
            save_code=True,
        )#mode = 'disabled')

        run_dir = mkdir(get_runs_dir() / f"F1TenthWayPoint_JAXRL" / name) 
        ckpt_dir = mkdir(run_dir / "ckpts")
        
       
        train_env = F1TenthWayPoint()
        eval_env = F1TenthWayPoint()
        state_example = np.zeros([train_env.nx])
        obs_example = np.zeros([train_env.nobs])
        act_example = np.zeros([train_env.nu])
       
        seed = kwargs.pop('seed')

        if self.agent is None:
            self.agent = algo(seed, str(ckpt_dir), obs_example, act_example, **kwargs) 

        replay_buffer = ReplayBuffer(obs_example, act_example, self.replay_buffer_size)

        eval_returns = []
        state = train_env.reset(mode='train')
        observation = train_env.get_obs(state)
        tot_step = 0
        for i in tqdm.tqdm(range(1, self.max_steps + 1),
                        smoothing=0.1,
                        disable=not self.tqdm):
            if i >= self.start_training:
                action = train_env.get_expert_control()
            else:
                action = self.agent.sample_actions(observation)
            next_state = train_env.step(state, action)
            next_observation = train_env.get_obs(next_state)
            reward = - train_env.l(next_state, action)
            done = train_env.should_reset()
            tot_step += 1
            if not done and tot_step <= 1e3:
                mask = 1.0
            else:
                mask = 0.0
                tot_step = 0
                done = True

            replay_buffer.insert(observation, action, reward, mask, float(done),
                                next_observation)
            
            observation = next_observation
            state = next_state

            if done:
                state = train_env.reset(mode='train')
                observation = train_env.get_obs(state)
    
            if i >= self.start_training:
                for _ in range(self.updates_per_step):
                    batch = replay_buffer.sample(self.batch_size)
                    update_info = self.agent.update(batch)

                if i % self.log_interval == 0:
                    for k, v in update_info.items():
                        wandb.log({f'training/{k}': v}, i)
                    

                if i % self.eval_interval == 0:
                    eval_stats = evaluate(self.agent, eval_env, self.eval_episodes)

                    for k, v in eval_stats.items():
                        wandb.log({f'evaluation/average_{k}s': v}, i)
                    

                    print(eval_stats)
                    eval_returns.append((i, eval_stats['length'], eval_stats['cost'], eval_stats['err']))
    
                    self.agent.save(i)
                    np.savetxt(os.path.join(ckpt_dir, f'{self.seed}.txt'), eval_returns, fmt=['%d', '%d', '%.1f', '%.1f'])
                
                    