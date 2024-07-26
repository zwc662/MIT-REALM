import time
import gym
import numpy as np
import yaml
import os

import argparse
from argparse import Namespace

from f110_gym.envs.base_classes import Integrator

from sysid.waypoint_follow import PurePursuitPlanner
from sysid.src.model import BicycleModel
from sysid.src.constants import Logging_Level


import logging
logging.basicConfig(
    level=Logging_Level.INFO.value , 
    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

 
import wandb
# Use wandb-core
wandb.require("core")
wandb.login()


def render_callback(env_renderer, planner):
    # custom extra drawing function

    e = env_renderer

    # update camera to follow car
    x = e.cars[0].vertices[::2]
    y = e.cars[0].vertices[1::2]
    top, bottom, left, right = max(y), min(y), min(x), max(x)
    e.score_label.x = left
    e.score_label.y = top - 700
    e.left = left - 800
    e.right = right + 800
    e.top = top + 800
    e.bottom = bottom - 800

    planner.render_waypoints(env_renderer) 

def collect_data_call_back(self):
    wandb.log({
            'x': self.data['Ys']['xs'][-1],
            'y': self.data['Ys']['ys'][-1],
            'poses_theta': self.data['Xs']['poses_theta'][-1],
            'velocity_x': self.data['Xs']['velocity_x'][-1],
            'velocity_y': self.data['Xs']['velocity_y'][-1],
            'pred_x': self.data['Ys']['xs'][-1],
            'pred_y': self.data['Ys']['ys'][-1],
            'res_x': self.data['Ys']['xs'][-2] if len(self.data['Ys']['xs']) > 1 else 0,
            'res_y': self.data['Ys']['ys'][-2] if len(self.data['Ys']['ys']) > 1 else 0
        }
    )
    

def update_models_call_back(info):
     wandb.log(info)

def main():
    """
    main entry point
    """
    

    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, required=True, help='Path to the map without extensions')
    args = parser.parse_args()



    work = {'mass': 3.463388126201571, 'lf': 0.15597534362552312, 'tlad': 0.82461887897713965, 'vgain': 1.375}#0.90338203837889}
    
    with open('config_example_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)


    planner = PurePursuitPlanner(conf, (0.17145+0.15875)) #FlippyPlanner(speed=0.2, flip_every=1, steer=10)
    bm = BicycleModel(planner.wheelbase, 
                      num_epochs = 10,
                      num_samples = 100,
                      algo = args.algo,
                      )



    wandb.init(
    #mode="disabled",
    # Set the project where this run will be logged
    project="basic-intro",
    # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
    name="gp_naive_100",
    # Track hyperparameters and run metadata
    )

 
    
    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1, timestep=0.01, integrator=Integrator.RK4)
    #env.add_render_callback(render_callback)
    
    obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
    #env.render()
    

    laptime = 0.0
    start = time.time()

    step = 0
    while not done:
        step += 1 
        speed, steer = planner.plan(obs, work)
        noise = np.random.normal([2]) * 0.1
        obs, step_reward, done, info = env.step(np.array([[steer + noise[0], speed + noise[0]]]))
        bm.collect_data(obs, steer, call_back = collect_data_call_back, logger = logger)
        
        laptime += step_reward
        #env.render(mode='human')
        if step > 50 and step % 100 == 1:
            print(f'{step=}') #logger.log(Logging_Level.INFO.value, f'Step {step}')
            bm.update_model(call_back = update_models_call_back, logger = logger)

        step += 1
        
    logger.log(Logging_Level.INFO.value, f'Sim elapsed time: {laptime} | Real elapsed time: {time.time()-start}')

if __name__ == '__main__':
    main()  