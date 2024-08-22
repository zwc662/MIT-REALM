import pytest
import os
import sys
 
import numpy as np
import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.random as jr

 
from efppo.rl.collector import CollectorCfg, Collector
from efppo.utils.tfp import tfd
from test_f110 import task



@pytest.fixture
def collect_cfg():
    return CollectorCfg(
        n_envs = 1, 
        rollout_T = 4, 
        mean_age = 10000, 
        max_T = 10000
        )

@pytest.fixture
def collector(task, collect_cfg):
    key = jr.PRNGKey(0)
    return Collector.create(key, task, collect_cfg)

@pytest.fixture
def get_pol():
    def help(obs_pol, state_z):
        return tfd.Bernoulli(logits=obs_pol[jnp.array([-2, -1])])
 
    return help

def test_collector(collector, get_pol):
    return collector.collect_batch_iteratively(get_pol, 0.99, 0, 10)


def test_step(task):
    obs0, reward0, done0, info0 = task.reset(mode='train')


    control = np.ones([2])
    obs, reward, done, info = task.step(None, np.ones([2]))
    print(obs, reward, done, info)

def test_scan(task):
    # Define the step function for jax.lax.scan
  
    def step_fn(carry, _):
        control = np.ones([2])
        obs, reward, done, info = task.step(carry, control)
        return obs, (obs, reward, done, info)
    obs0, reward0, done0, info0 = task.reset(mode='train')
    print(obs0, reward0)
    final_obs, (obs_seq, reward_seq, done_seq, info_seq) = lax.scan(step_fn, obs0, None, length=10)
    print(final_obs, (obs_seq, reward_seq, done_seq, info_seq))



 
   
if __name__ == "__main__":
    pytest.main()