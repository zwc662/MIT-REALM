import pytest
import os
import sys
sys.path.append(
    os.path.join(
        os.path.dirname(__file__).split('test')[0],
        'src'
    )
)

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

 
   
if __name__ == "__main__":
    pytest.main()