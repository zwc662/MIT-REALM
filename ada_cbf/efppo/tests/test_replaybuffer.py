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
from test_f110_collector import task, collect_cfg, collector, get_pol

from efppo.utils.replay_buffer import Experiences, ReplayBuffer



def test_replaybuffer(collector, get_pol):
    rb = ReplayBuffer.create(key = jr.PRNGKey(0), capacity = 10)
    new_collector, rollouts = collector.collect_batch_iteratively(get_pol, 0.99, 0, 10)
    new_rb = rb.insert(rollouts)
   
if __name__ == "__main__":
    pytest.main()