
from abc import ABC, abstractmethod
from typing import Optional, Callable, Tuple, Dict, Any, List

import math

import jax
import jax.numpy as jnp 



import optax

import pickle
  
from sysid.src.constants import Logging_Level
from sysid.src.nn import NN_Learner
from sysid.src.gp import GP_Learner

from sklearn.model_selection import train_test_split



class DynamicalModel(ABC):
    def __init__(self):
        self.X = None
        self.Y = None 
        self.clear_cache()

    @abstractmethod
    def clear_cache(self): ...
    
    @abstractmethod
    def collect_data(self, data): ...

    @abstractmethod
    def update_model(self, *args, **kwargs): ...
 
        


### Bicycle Model
 

"""
Bicycle Model
"""
 
 
class BicycleModel(DynamicalModel):
    def __init__(self, 
                wheelbase: float, 
                dt: float = 0.01,   
                rng: int = 0,
                algo: str = 'nn',
                **kwargs
    ):
        super(BicycleModel).__init__()
        
        
        self.wheelbase = wheelbase
        self.dt = dt 

        self.avg_loss = None
        self.mse_x = None
        self.mse_y = None
        self.mean_preds = None

        model_key, self.random_key = jax.random.split(jax.random.PRNGKey(rng))
        
        self.model = None
        if algo == 'nn':
            self.model = NN_Learner(random_key = model_key, **kwargs)
        elif algo == 'gp':
            self.model = GP_Learner(random_key = model_key, num_samples = kwargs['num_samples'])
         
        self.clear_cache()
        self.X = None
        self.Y = None

    def clear_cache(self):
        self.data = {'Xs': {'xs': [], 'ys': [], 'poses_theta': [], 'velocity_x': [], 'velocity_y': [], 'steering_angle': []},
                        'Ys': {'xs': [], 'ys': []}}
        
    def collect_data(self, obs: Dict[str, List[float]], steering_angle: float, call_back: Optional[Callable] = None, logger: Any = None):
        logger.log(Logging_Level.STASH.value, f'obs = {obs}')

        pose_x: float = jnp.array(obs['poses_x'])
        pose_y: float = jnp.array(obs['poses_y'])
        pose_theta: float = jnp.array(obs['poses_theta'])
        velocity_x: float = jnp.array(obs['linear_vels_x'])
        velocity_y: float = jnp.array(obs['linear_vels_y'])
        
        logger.log(Logging_Level.STASH.value, f'velocity_x: {velocity_x}')
        logger.log(Logging_Level.STASH.value, f'velocity_y: {velocity_y}')

        beta = jnp.arctan(0.5 * jnp.tan(steering_angle))
        velocity = jnp.sqrt(jnp.square(velocity_x) + jnp.square(velocity_y))

        logger.log(Logging_Level.STASH.value, f'velocity: {velocity} + jnp.cos(pose_theta + beta): {jnp.cos(pose_theta + beta)}')
        dx = velocity * jnp.cos(pose_theta + beta) * self.dt
        dy = velocity * jnp.sin(pose_theta + beta) * self.dt
        dtheta = (velocity / self.wheelbase) * jnp.sin(beta) * self.dt

        # Collect ijnput
        self.data['Xs']['xs'].append(pose_x)
        self.data['Xs']['ys'].append(pose_y)
        self.data['Xs']['poses_theta'].append(pose_theta)
        self.data['Xs']['velocity_x'].append(velocity_x)
        self.data['Xs']['velocity_y'].append(velocity_y)
        
        self.data['Xs']['steering_angle'].append(jnp.array([steering_angle]))
        
        # Correct Y in the previous step based on true state variables
        if len(self.data['Ys']['xs']) > 0:
            self.data['Ys']['xs'][-1] = pose_x - self.data['Ys']['xs'][-1]
        if len(self.data['Ys']['ys']) > 0:
            self.data['Ys']['ys'][-1] = pose_y - self.data['Ys']['ys'][-1]

        # Predict the next Y
        self.data['Ys']['xs'].append(pose_x + dx)
        self.data['Ys']['ys'].append(pose_y + dy)

        logger.log(Logging_Level.STASH.value, f"len(self.data['Ys']['xs']) = {len(self.data['Ys']['xs'])}")
        logger.log(Logging_Level.STASH.value, f"len(self.data['Xs']['xs']) = {len(self.data['Xs']['xs'])}")
        
        # Synch w/ wandb
        call_back(self)

 
    
    def prepare_data(self, logger: Any = None):
        logger.log(Logging_Level.STASH.value, 
                   f"self.data['Xs']['poses_theta'][0] = {self.data['Xs']['poses_theta'][0]}, \
                    self.data['Xs']['velocity_x'][0] = {self.data['Xs']['velocity_x']}, \
                        self.data['Xs']['velocity_y'][0] = {self.data['Xs']['velocity_y']}, \
                            self.data['Xs']['steering_angle']] = {self.data['Xs']['steering_angle']}"
                            )
        X = jnp.hstack((
            jnp.stack(self.data['Xs']['xs']),
            jnp.stack(self.data['Xs']['ys']),
            jnp.stack(self.data['Xs']['poses_theta']),
            jnp.stack(self.data['Xs']['velocity_x']),
            jnp.stack(self.data['Xs']['velocity_y']),
            jnp.stack(self.data['Xs']['steering_angle'])
            ))
        Y = jnp.hstack((
            jnp.stack(self.data['Ys']['xs']),
            jnp.stack(self.data['Ys']['ys'])
            ))
        
        logger.log(Logging_Level.STASH.value, f'X.shape = {X.shape}, Y.shape = {Y.shape}')
        
        if self.X is None and self.Y is None:
            self.X = X[1:]
            self.Y = Y[1:]
        else:
            self.X = jnp.vstack((self.X, X))
            self.Y = jnp.vstack((self.Y, Y))
        
        logger.log(Logging_Level.STASH.value, f'self.X.shape = {self.X.shape}, self.Y.shape = {self.Y.shape}')
        

        self.clear_cache()
   


    def update_model(self, 
                      call_back: Optional[Callable] = None,
                      logger: Any = None
                      ):
        self.prepare_data(logger)
        info = self.model.update(self.X, self.Y, logger)
        call_back(info)       

    
    