from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional, Callable

import jax
import jax.numpy as jnp 
jax.config.update('jax_platform_name', 'cpu')

import numpy as np


import pickle
  
from sysid.src.constants import Logging_Level
#from sysid.src.learner.nn import NN_Learner
from sysid.src.learner.gp import GP_Learner
from sysid.src.learner.safeopt import SafeOpt_Learner
from sysid.src.learner.lin import Linear_Learner, NN_Learner, Poly_Learner
from sysid.src.learner.nn_torch import SN_NN_Learner


from sklearn.model_selection import train_test_split

class Learner:
    def __init__(self, algo: str, rng: Any, update_interval: int = 100, **kwargs):
        
        random_key, self.random_key = jax.random.split(jax.random.PRNGKey(rng))
         
        self.init_learner(algo, random_key, **kwargs)

        self.init_cache()
        self.X = None
        self.Y = None
    
    def init_learner(self, algo: str, random_key: Any, **kwargs): 
        self.algo = algo
        if self.algo == 'nn':
            self.model = NN_Learner(random_key = random_key) #, **kwargs)
        elif self.algo == 'gp':
            self.model = GP_Learner(random_key = random_key, num_samples = kwargs['num_samples'])
        elif self.algo == 'safeopt':
            self.model = SafeOpt_Learner(random_key = random_key, num_samples = kwargs['num_samples'])
        elif self.algo == 'lin':
            self.model = Linear_Learner(random_key = random_key)
        elif self.algo == 'poly':
            self.model = Poly_Learner(random_key = random_key)
        elif self.algo == 'sn':
            self.model = SN_NN_Learner(random_key = random_key, num_epochs = 100)
         

    def init_cache(self):
        self.buffer = {'Xs': {'xs': [], 'ys': [], 'poses_theta': [], 'velocity_x': [], 'velocity_y': [], 'accelerate': [], 'steering': []},
                        'Ys': {'xs': [], 'ys': [], 'pred_xs': [], 'pred_ys': []}}
    
    def clear_buffer(self, keep_last: int):
        for k, v in self.buffer['Xs'].items():
            self.buffer['Xs'][k] = v[-keep_last:]
        for k, v in self.buffer['Ys'].items():
            self.buffer['Ys'][k] = v[-keep_last:]

    
    def collect_data_one_step(
            self,  
            obs: Dict[str, Any], 
            #pred_accelerate: Any,
            #pred_steering: Any,
            pred_nxt_obs: Dict[str, Any],
            accelerate: Any, 
            steering: Any, 
            call_back: Optional[Callable] = None, 
            logger: Any = None
            ):
        logger.log(Logging_Level.STASH.value, f'obs = {obs}')

        pose_x: float = np.array(obs['poses_x'])
        pose_y: float = np.array(obs['poses_y'])
        pose_theta: float = np.array(obs['poses_theta'])
        velocity_x: float = np.array(obs['linear_vels_x'])
        velocity_y: float = np.array(obs['linear_vels_y'])

        #pred_accelerate: float = np.array([pred_accelerate])
        #pred_steering: float = np.array([pred_steering])

        accelerate: float = np.array([accelerate])
        steering: float = np.array([steering])

        nxt_pose_x: float = np.array(pred_nxt_obs['poses_x'])
        nxt_pose_y: float = np.array(pred_nxt_obs['poses_y'])


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

        # Predict the next state with learning model
        try:
            dx_dy = self.pred(obs, accelerate, steering, logger = logger)
        except:
            #print("Make no prediction yet")
            dx_dy = [0, 0]


        self.buffer['Ys']['pred_xs'].append(nxt_pose_x + dx_dy[0])
        self.buffer['Ys']['pred_ys'].append(nxt_pose_y + dx_dy[1])

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
        X = np.hstack((
            np.stack(self.buffer['Xs']['xs']),
            np.stack(self.buffer['Xs']['ys']),
            np.stack(self.buffer['Xs']['poses_theta']), 
            np.stack(self.buffer['Xs']['velocity_x']),
            np.stack(self.buffer['Xs']['velocity_y']),
            np.stack(self.buffer['Xs']['accelerate']),
            np.stack(self.buffer['Xs']['steering'])
            ))[:-1]
        Y = np.hstack((
            np.stack(np.asarray(self.buffer['Ys']['xs'][1:]) - np.asarray(self.buffer['Xs']['xs'][:-1])),
            np.stack(np.asarray(self.buffer['Ys']['ys'][1:]) - np.asarray(self.buffer['Xs']['ys'][:-1]))
            ))
        if logger.level == Logging_Level.TEST.value:
            Y = Y.at[:, 0].set(np.cos(X[:, 2] + X[:, 3]))
            Y = Y.at[:, 1].set(np.sin(X[:, 2] - X[:, 3]))
        
        assert X.shape[0] == Y.shape[0], f'X.shape = {X.shape}, Y.shape = {Y.shape}'

        if self.X is None and self.Y is None:
            self.X = X 
            self.Y = Y 
        else:
            self.X = np.vstack((self.X, X))
            self.Y = np.vstack((self.Y, Y))
        
        logger.log(Logging_Level.STASH.value, f'self.X.shape = {self.X.shape}, self.Y.shape = {self.Y.shape}')
        

        self.clear_buffer(1)
   


    def update_model(self, 
                      call_back: Optional[Callable] = None,
                      logger: Any = None
                      ):
        if self.algo == 'bm':
            return 
        self.prepare_data(logger)
        info = self.model.update(self.X, self.Y, logger)
        call_back(info)       

    def pred(self, obs, accelerate, steering, logger = None):
        if self.algo == 'bm':
            return np.zeros([2])
        pose_theta: float = np.array(obs['poses_theta'])
        velocity_x: float = np.array(obs['linear_vels_x'])
        velocity_y: float = np.array(obs['linear_vels_y'])

        accelerate: float = np.array([accelerate])
        steering: float = np.array([steering])

        x = np.concatenate((pose_theta, velocity_x, velocity_y, accelerate, steering)).reshape(1, -1)
        
        #print(f"{x=}, {x.shape=}")
        for x_ in x:
            if np.isnan(x_).any():
                return None
        means, vars = self.model.eval(x, logger)

        means = means.reshape(2)
 
        return means