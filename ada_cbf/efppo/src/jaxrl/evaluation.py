from typing import Dict

import flax.linen as nn
import gym
import numpy as np
import jax.numpy as jnp
import jax.tree_util as jtu


def evaluate(agent, env: gym.Env, num_episodes: int) -> Dict[str, float]:
    stats = {'cost': [], 'length': [], 'err': []}
    for _ in range(num_episodes):
        tot_cost = 0
        tot_length = 0
        tot_err = 0
        done = False
        observation = env.reset()
        tot_step = 0
        while not done and tot_step <= 1e3:
            action = agent.sample_actions(observation, temperature=0.0)
            observation = env.step(observation, action)
            cost = env.l(observation, action)
            expert_control = env.get_expert_control(observation, action)
            err = np.square(np.asarray(action).flatten() - np.asarray(expert_control).flatten()).sum()
            done = env.should_reset()
            tot_err += err
            tot_cost += cost
            tot_length += 1
            tot_step += 1
        stats['cost'].append(tot_cost / tot_length)
        stats['length'].append(tot_length)
        stats['err'].append(tot_err / tot_length)

    return {k: jnp.mean(jnp.stack(v)) for k, v in stats.items()}
