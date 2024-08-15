
from abc import ABC, abstractmethod
from typing import Optional, Callable, Tuple, Dict, Any, List

import math

import jax
import jax.numpy as np 



import optax

import pickle
  
from sysid.src.constants import Logging_Level 
 
 

### Bicycle Model
 

"""
Bicycle Model
"""
 
 
class BicycleModel(object):
    def __init__(self, 
                d: float = 0.01,
                m: float = 1.0, 
                lf: float = 1.0,
                lr: float = 1.0,
                steer_coef: float = 0.5,
                dt: float = 0.01,  
                rng: int = 0,
                algo: str = 'nn',
                **kwargs
    ):
        """
        Initialize the bicycle model.
        :param wheelbase: wheelbase
        :param d: drag coefficient
        :param m: mass of the bicycle
        :param lf (lr): distance between front (rear) axle and gravity center
        """
          
        super(BicycleModel).__init__()
        
        
        self.d = d
        self.m = m
        self.lf = lf
        self.lr = lr


        self.dt = dt 

             
    def step(self, obs: Dict[str, List[float]], accelerate: float, steering_angle: float, call_back: Optional[Callable] = None, logger: Any = None):
        logger.log(Logging_Level.STASH.value, f'obs = {obs}')

        pose_x: float = np.array(obs['poses_x'])
        pose_y: float = np.array(obs['poses_y'])
        pose_theta: float = np.array(obs['poses_theta'])
        velocity_x: float = np.array(obs['linear_vels_x'])
        velocity_y: float = np.array(obs['linear_vels_y'])

        accelerate: float = np.array([accelerate])
        steering_angle: float = np.array([steering_angle])


        logger.log(Logging_Level.STASH.value, f'velocity_x: {velocity_x}')
        logger.log(Logging_Level.STASH.value, f'velocity_y: {velocity_y}')


        
        f_d = self.d * (np.square(velocity_x) + np.square(velocity_y))
        a_net = (accelerate - f_d) / self.m

        beta = np.arctan(self.lr / (self.lr + self.lf) * np.tan(steering_angle))
        velocity = np.sqrt(np.square(velocity_x) + np.square(velocity_y))

        logger.log(Logging_Level.STASH.value, f'{velocity=}, {beta =}')
        
        dx = velocity * np.cos(pose_theta + beta) * self.dt
        dy = velocity * np.sin(pose_theta + beta) * self.dt

        dtheta = velocity / (self.lr + self.lf) * np.cos(beta) * np.tan(steering_angle) * self.dt
 

        return {
            'poses_x': pose_x + dx, 
            'poses_y': pose_y + dy, 
            'poses_theta': pose_theta + dtheta, 
            'linear_vels_x': (velocity + a_net * self.dt) * np.cos(pose_theta + dtheta),
            'linear_vels_y': (velocity + a_net * self.dt) * np.sin(pose_theta + dtheta),
            }
    
    
    