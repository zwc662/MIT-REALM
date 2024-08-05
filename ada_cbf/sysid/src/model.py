
from abc import ABC, abstractmethod
from typing import Optional, Callable, Tuple, Dict, Any, List

import math

import jax
import jax.numpy as jnp 



import optax

import pickle
  
from sysid.src.constants import Logging_Level 
 
 

### Bicycle Model
 

"""
Bicycle Model
"""
 
 
class BicycleModel(object):
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

             
    def step(self, obs: Dict[str, List[float]], accelerate: float, steering_angle: float, call_back: Optional[Callable] = None, logger: Any = None):
        logger.log(Logging_Level.STASH.value, f'obs = {obs}')

        pose_x: float = jnp.array(obs['poses_x'])
        pose_y: float = jnp.array(obs['poses_y'])
        pose_theta: float = jnp.array(obs['poses_theta'])
        velocity_x: float = jnp.array(obs['linear_vels_x'])
        velocity_y: float = jnp.array(obs['linear_vels_y'])

        accelerate: float = jnp.array([accelerate])
        steering_angle: float = jnp.array([steering_angle])


        logger.log(Logging_Level.STASH.value, f'velocity_x: {velocity_x}')
        logger.log(Logging_Level.STASH.value, f'velocity_y: {velocity_y}')

        beta = jnp.arctan(0.5 * jnp.tan(steering_angle))
        velocity = jnp.sqrt(jnp.square(velocity_x) + jnp.square(velocity_y))

        logger.log(Logging_Level.STASH.value, f'{velocity=}, {beta =}')
        
        dx = velocity * jnp.cos(pose_theta + beta) * self.dt
        dy = velocity * jnp.sin(pose_theta + beta) * self.dt

        dtheta = (velocity / self.wheelbase) * jnp.sin(beta) * self.dt
 

        return {
            'poses_x': pose_x + dx, 
            'poses_y': pose_y + dy, 
            'poses_theta': pose_theta + dtheta, 
            'linear_vels_x': (velocity + accelerate * self.dt) * jnp.cos(pose_theta + dtheta),
            'linear_vels_y': (velocity + accelerate * self.dt) * jnp.sin(pose_theta + dtheta),
            }
    
    
    