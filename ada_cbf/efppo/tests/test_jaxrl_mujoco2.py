import os
import random
import time

import numpy as np
import tqdm
from absl import app, flags 

import jax.numpy as jnp

from typing import Optional, Union, Dict

from jaxrl.agents import (AWACLearner, DDPGLearner, REDQLearner, SACLearner,
                          SACV1Learner)
from jaxrl.datasets import ReplayBuffer 
 

import pytest

from dataclasses import dataclass, asdict

import gymnasium as gym

import wandb

def evaluate(agent, env: gym.Env, num_episodes: int) -> Dict[str, float]:
    stats = {'avg_return': [], 'length': [], 'tot_return': []}
    for _ in range(num_episodes):
        tot_return = 0
        tot_length = 0
        done = False 
        truncated = False
        observation, info = env.reset()
        while not done and not truncated and tot_length <= 1e3:
            action = agent.sample_actions(observation, temperature=0.0).clip(-1, 1)
            observation, reward, done, truncated, info = env.step(action)
            
            tot_return += reward
            tot_length += 1
             
        stats['tot_return'].append(tot_return)
        stats['avg_return'].append(tot_return / tot_length)
        stats['length'].append(tot_length) 

    return {k: jnp.mean(jnp.stack(v)) for k, v in stats.items()} 

@dataclass
class Cfg: 
    env: str = 'Hopper-v4'
    seed: int = 42
    eval_episodes: int  = 10
    log_interval: int = int(1e3)        # ('Logging interval.')
    eval_interval: int = int(5e3)       # ('Eval interval.')
    batch_size: int = 256        # ('Mini batch size.')
    updates_per_step: int = 1        # ('Gradient updates per step.')
    max_steps: int = int(1e7)        # ('Number of training steps.')
    start_training: int = int(1e4)      #'Number of training steps to start training.')
    tqdm: bool = True
    replay_buffer_size: int = int(1e7)
    
@pytest.fixture
def test_file_dir(request):
    return request.path.parent

def test_train(test_file_dir):
    kwargs = asdict(Cfg())
    
    wandb.init(
        project='jaxrl',
        entity='zwc662',
        name='test_jaxrl_mujoco2_sac',
        monitor_gym=True,
        save_code=True,
    )#mode = 'disabled')
  
    env = kwargs.pop('env')
    train_env = gym.make(env)

    obs_example = np.zeros(train_env.observation_space.shape)
    act_example = np.zeros(train_env.action_space.shape)


    run_dir = test_file_dir / 'test_jaxrl_mujoco' / env
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)


    eval_env = gym.make(env) 
    
    seed = kwargs.pop('seed')

    np.random.seed(seed)
    random.seed(seed)


    start_training = kwargs.pop('start_training')
    updates_per_step = kwargs.pop('updates_per_step')
    batch_size = kwargs.pop('batch_size')
    eval_episodes = kwargs.pop('eval_episodes')
    max_steps = kwargs.pop('max_steps')
    log_interval = kwargs.pop('log_interval')
    eval_interval = kwargs.pop('eval_interval')

    agent = SACLearner(seed,
                        obs_example,
                        act_example, **kwargs)
    
    replay_buffer_size = kwargs.pop('replay_buffer_size')
    replay_buffer = ReplayBuffer(train_env.observation_space, train_env.action_space,
                                 replay_buffer_size)

    eval_returns = []
    observation, info = train_env.reset()
    done = False
    truncated = False
    for i in tqdm.tqdm(range(1, max_steps + 1),
                       smoothing=0.1,
                       disable=not tqdm):
        if i < start_training:
            action = train_env.action_space.sample()
        else:
            action = agent.sample_actions(observation)
        next_observation, reward, done, truncated, info = train_env.step(action)

        if not done and not truncated:
            mask = 1.0
        else:
            mask = 0.0

        replay_buffer.insert(observation, action, reward, mask, float(done),
                             next_observation)
        observation = next_observation

        if done:
            observation, info = train_env.reset()
            done = truncated = False
            
        if i >= start_training:
            for _ in range(updates_per_step):
                batch = replay_buffer.sample(batch_size)
                update_info = agent.update(batch)

            if i % log_interval == 0:
                for k, v in update_info.items():
                    wandb.log({f'training/{k}': v}, i)


            if i % eval_interval == 0:
                eval_stats = evaluate(agent, eval_env, eval_episodes)

                for k, v in eval_stats.items():
                    wandb.log({f'evaluation/average_{k}s': v}, i)
                

                print(eval_stats)
                eval_returns.append((i, eval_stats['length'], eval_stats['avg_return'], eval_stats['tot_return']))
 
                np.savetxt(os.path.join(ckpt_dir, f'{seed}.txt'), eval_returns, fmt=['%d', '%d', '%.1f', '%.1f'])
            


if __name__ == '__main__':
    app.run(main)