from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional, Callable

import jax
import jax.numpy as jnp

import pickle
  
from sysid.src.constants import Logging_Level
from sysid.src.learner.nn import NN_Learner
from sysid.src.learner.gp import GP_Learner
from sysid.src.learner.safeopt import SafeOpt_Learner
from sysid.src.learner.lin import Linear_Learner, Poly_Learner


from sklearn.model_selection import train_test_split

class Learner:
    def __init__(self, algo: str, rng: Any, update_interval: int = 100, **kwargs):
        
        random_key, self.random_key = jax.random.split(jax.random.PRNGKey(rng))
         
        self.init_learner(algo, random_key, **kwargs)

        self.init_cache()
        self.X = None
        self.Y = None
    
    def init_learner(self, algo: str, random_key: Any, **kwargs): 
        if algo == 'nn':
            self.model = NN_Learner(random_key = random_key, **kwargs)
        elif algo == 'gp':
            self.model = GP_Learner(random_key = random_key, num_samples = kwargs['num_samples'])
        elif algo == 'safeopt':
            self.model = SafeOpt_Learner(random_key = random_key, num_samples = kwargs['num_samples'])
        elif algo == 'lin':
            self.model = Linear_Learner(random_key = random_key)
        elif algo == 'poly':
            self.model = Poly_Learner(random_key = random_key)

    def init_cache(self):
        self.buffer = {'Xs': {'xs': [], 'ys': [], 'poses_theta': [], 'velocity_x': [], 'velocity_y': [], 'accelerate': [], 'steering': []},
                        'Ys': {'xs': [], 'ys': []}}
    
    def clear_buffer(self, keep_last: int):
        for k, v in self.buffer['Xs'].items():
            self.buffer['Xs'][k] = v[-keep_last:]
        for k, v in self.buffer['Ys'].items():
            self.buffer['Ys'][k] = v[-keep_last:]

    
    def collect_data_one_step(
            self, 
            obs: Dict[str, Any], 
            accelerate: Any, 
            steering: Any, 
            pred: Dict[str, Any],
            call_back: Optional[Callable] = None, 
            logger: Any = None
            ):
        logger.log(Logging_Level.STASH.value, f'obs = {obs}')

        pose_x: float = jnp.array(obs['poses_x'])
        pose_y: float = jnp.array(obs['poses_y'])
        pose_theta: float = jnp.array(obs['poses_theta'])
        velocity_x: float = jnp.array(obs['linear_vels_x'])
        velocity_y: float = jnp.array(obs['linear_vels_y'])

        accelerate: float = jnp.array([accelerate])
        steering: float = jnp.array([steering])

        nxt_pose_x: float = jnp.array(pred['poses_x'])
        nxt_pose_y: float = jnp.array(pred['poses_y'])


        # Collect input
        self.buffer['Xs']['xs'].append(pose_x)
        self.buffer['Xs']['ys'].append(pose_y)

        self.buffer['Xs']['poses_theta'].append(pose_theta)

        self.buffer['Xs']['velocity_x'].append(velocity_x)
        self.buffer['Xs']['velocity_y'].append(velocity_y)
        
        self.buffer['Xs']['accelerate'].append(accelerate)
        self.buffer['Xs']['steering'].append(steering)
        
        # Correct Y in the previous step based on true state variables
        self.buffer['Ys']['xs'].append(nxt_pose_x)
        self.buffer['Ys']['ys'].append(nxt_pose_y)

        logger.log(Logging_Level.STASH.value, f"len(self.buffer['Ys']['xs']) = {len(self.buffer['Ys']['xs'])}")
        logger.log(Logging_Level.STASH.value, f"len(self.buffer['Xs']['xs']) = {len(self.buffer['Xs']['xs'])}")
        
        # Synch w/ wandb
        if call_back is not None:
            call_back(self)

 
    
    def prepare_data(self, logger: Any = None):
        
        logger.log(Logging_Level.STASH.value, 
                   f"self.buffer['Xs']['poses_theta'][0] = {self.buffer['Xs']['poses_theta'][0]}, \
                    self.buffer['Xs']['velocity_x'][0] = {self.buffer['Xs']['velocity_x']}, \
                        self.buffer['Xs']['velocity_y'][0] = {self.buffer['Xs']['velocity_y']}, \
                        self.buffer['Xs']['accelerate'][0] = {self.buffer['Xs']['accelerate']}, \
                                self.buffer['Xs']['steering']] = {self.buffer['Xs']['steering']}"
                            )
        X = jnp.hstack((
            jnp.stack(self.buffer['Xs']['xs']),
            jnp.stack(self.buffer['Xs']['ys']),
            jnp.stack(self.buffer['Xs']['poses_theta']), 
            jnp.stack(self.buffer['Xs']['velocity_x']),
            jnp.stack(self.buffer['Xs']['velocity_y']),
            jnp.stack(self.buffer['Xs']['accelerate']),
            jnp.stack(self.buffer['Xs']['steering'])
            ))[:-1]
        Y = jnp.hstack((
            jnp.stack(jnp.asarray(self.buffer['Ys']['xs'][1:]) - jnp.asarray(self.buffer['Xs']['xs'][:-1])),
            jnp.stack(jnp.asarray(self.buffer['Ys']['ys'][1:]) - jnp.asarray(self.buffer['Xs']['ys'][:-1]))
            ))
        if logger.level == Logging_Level.TEST.value:
            Y = Y.at[:, 0].set(jnp.cos(X[:, 2] + X[:, 3]))
            Y = Y.at[:, 1].set(jnp.sin(X[:, 2] - X[:, 3]))
        
        assert X.shape[0] == Y.shape[0], f'X.shape = {X.shape}, Y.shape = {Y.shape}'

        if self.X is None and self.Y is None:
            self.X = X 
            self.Y = Y 
        else:
            self.X = jnp.vstack((self.X, X))
            self.Y = jnp.vstack((self.Y, Y))
        
        logger.log(Logging_Level.STASH.value, f'self.X.shape = {self.X.shape}, self.Y.shape = {self.Y.shape}')
        

        self.clear_buffer(1)
   


    def update_model(self, 
                      call_back: Optional[Callable] = None,
                      logger: Any = None
                      ):
        self.prepare_data(logger)
        info = self.model.update(self.X, self.Y, logger)
        call_back(info)       

    def pred(self, obs, accelerate, steering, logger = None):
        pose_theta: float = jnp.array(obs['poses_theta'])
        velocity_x: float = jnp.array(obs['linear_vels_x'])
        velocity_y: float = jnp.array(obs['linear_vels_y'])

        accelerate: float = jnp.array([accelerate])
        steering: float = jnp.array([steering])

        x = jnp.concatenate((pose_theta, velocity_x, velocity_y, accelerate, steering)).reshape(1, -1)
        
        #print(f"{x=}, {x.shape=}")
        means, vars = self.model.eval(x, logger)

        means = means.reshape(2)

        dx = means[0].item()
        dy = means[1].item()

        return dx, dy