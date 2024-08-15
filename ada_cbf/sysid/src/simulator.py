

import jax
import jax.numpy as np

from typing import List, Dict, Optional, Tuple


class Simulator(object):
    def __init__(self, learner, bm, planner, work):
        self.learner = learner
        self.bm = bm 
        self.planner = planner
        self.work = work

    def forward(self, obs, steps = 100, logger = None):
        """
        (Simulated) Step function for the gym env

        Args:
            action (np.ndarray(num_agents, 2))

        Returns:
            obs (dict): observation of the current step
            reward (float, default=self.timestep): step reward, currently is physics timestep
            done (bool): if the simulation is done
            info (dict): auxillary information dictionary
        """
        
        obss = [] 
        #acts = []
        cur_obs = {k: v for k, v in obs.items()}
        for _ in range(steps):
            # call simulation step

            #print(cur_obs)
            acc, steer = self.planner.plan(cur_obs, self.work)
 
            
            obss.append(cur_obs)
            #acts.append((acc + noise[0], steer + noise[0]))
            
            pred = self.bm.step(cur_obs, acc , steer  , None, logger = logger)
         
            dx_dy = self.learner.pred(cur_obs, acc , steer  , logger = logger)
            if dx_dy is None:
                break

            pred.update({
                'poses_x': pred['poses_x'] + dx_dy[0],
                'poses_y': pred['poses_y'] + dx_dy[1],
                }
            )
         
            cur_obs = {k: np.asarray([v]).reshape(1).tolist() for k, v in pred.items()}
        """
        def step_fn(carry, _):
            cur_obs, obss = carry
            acc, steer = self.planner.plan(cur_obs, self.work)
            noise = jnp.random.normal(key=jax.random.PRNGKey(0), shape=(2,)) * 0.1

            pred = self.bm.step(cur_obs, acc + noise[0], steer + noise[0], None, logger=logger)
            dx_dy = self.learner.pred(cur_obs, acc + noise[0], steer + noise[0], logger=logger)

            new_obs = {
                **pred,
                'poses_x': pred['poses_x'] + dx_dy[0],
                'poses_y': pred['poses_y'] + dx_dy[1],
            }

            new_obs = {k: jnp.asarray([v]).reshape(1).tolist() for k, v in new_obs.items()}
            obss = obss + [new_obs]
            
            return (new_obs, obss), None

        cur_obs = {k: v for k, v in obs.items()}
        obss = []
        
        (final_obs, final_obss), _ = lax.scan(step_fn, (cur_obs, obss), None, length=steps)
        
        return final_obss
        """
        return obss
