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
        tot_err = 0
        tot_length = 0
        done = False
        state = env.reset(mode='test')
        observation = env.get_obs(state) 
        while not done and tot_length <= 1e3:
            action = agent.sample_actions(observation, temperature=0.0).clip(-1, 1) * (env.ub - env.lb) / 2 + env.lb
            state = env.step(state, action)
            observation = env.get_obs(state)
            cost = env.l(state, action)
            expert_control = env.get_expert(state, action)
            err = np.square(np.asarray(action).flatten() - np.asarray(expert_control).flatten()).sum()
            done = env.should_reset()
            tot_err += err
            tot_cost += cost
            tot_length += 1
            

        stats['cost'].append(tot_cost / tot_length)
        stats['length'].append(tot_length)
        stats['err'].append(tot_err / tot_length)

    return {k: jnp.mean(jnp.stack(v)) for k, v in stats.items()}
