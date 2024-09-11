from typing import Dict

import flax.linen as nn
import gym
import jax.numpy as jnp
import jax.tree_util as jtu


def evaluate(agent, env: gym.Env, num_episodes: int) -> Dict[str, float]:
    stats = {'cost': [], 'length': []}
    for _ in range(num_episodes):
        tot_cost = 0
        tot_length = 0
        done = False
        observation = env.reset()
        while not done:
            action = agent.sample_actions(observation, temperature=0.0)
            observation = env.step(observation, action)
            cost = env.l(observation, action)
            done = env.should_reset()
            tot_cost += cost
            tot_length += 1
        stats['cost'].append(tot_cost / tot_length)
        stats['length'].append(tot_length)
 
    return {k: jnp.mean(jnp.stack(v)) for k, v in stats.items()}
