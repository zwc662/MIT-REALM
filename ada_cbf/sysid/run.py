import time
import gym
import numpy as np
import yaml
import os

import argparse
from argparse import Namespace

import jax
jax.config.update('jax_platform_name', 'cpu')

from f110_gym.envs.base_classes import Integrator

from sysid.waypoint_follow import PurePursuitPlanner
from sysid.src.model import BicycleModel
from sysid.src.constants import Logging_Level
from sysid.src.learner import Learner

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


import logging
logging.basicConfig(
    filename="log",
    filemode='w',
    format='%(name)s -  %(lineno)d - %(message)s')
logger = logging.getLogger()
logger.setLevel(Logging_Level.DEBUG.value)
 
import wandb
# Use wandb-core
wandb.require("core")
wandb.login()

wandb.init(
    mode="disabled",
    # Set the project where this run will be logged
    project="basic-intro",
    # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
    name="safeopt",
    # Track hyperparameters and run metadata
    )

 


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
            'x': self.buffer['Ys']['xs'][-1],
            'y': self.buffer['Ys']['ys'][-1],
            'poses_theta': self.buffer['Xs']['poses_theta'][-1],
            'velocity_x': self.buffer['Xs']['velocity_x'][-1],
            'velocity_y': self.buffer['Xs']['velocity_y'][-1],
            'accelerate': self.buffer['Xs']['accelerate'][-1],
            'steering': self.buffer['Xs']['steering'][-1],
            'pred_x': self.buffer['Ys']['xs'][-1],
            'pred_y': self.buffer['Ys']['ys'][-1],
            'res_x': self.buffer['Ys']['xs'][-2] if len(self.buffer['Ys']['xs']) > 1 else 0,
            'res_y': self.buffer['Ys']['ys'][-2] if len(self.buffer['Ys']['ys']) > 1 else 0
        }
    )
    

def update_models_call_back(info):
     wandb.log(info)

    
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

            
            noise = np.random.normal([2]) * 0.1
            
            obss.append(cur_obs)
            #acts.append((acc + noise[0], steer + noise[0]))
            
            pred = self.bm.step(obs, acc + noise[0], steer + noise[0], None, logger = logger)
            
            dx, dy = self.learner.pred(obs, acc + noise[0], steer + noise[0], logger = logger)

            cur_obs = pred
            
            cur_obs.update({
                'poses_x': pred['poses_x'] + dx,
                'poses_y': pred['poses_y'] + dy,
                }
            )

            cur_obs = {k: np.asarray([v]).reshape(1).tolist() for k, v in cur_obs.items()}

            
        return obss



# Function to draw the plot
def draw_plot(obs_points, all_sim_traj_points, filename):
    plt.figure()
    # Plot observation points in red
    plt.scatter(obs_points['x'], obs_points['y'], color='red', label='Observations')
    
    # Create a color map
    colors = cm.rainbow(np.linspace(0, 1, len(all_sim_traj_points)))
    
    # Plot each simulated trajectory with a different color
    for i, sim_traj_points in enumerate(all_sim_traj_points):
        plt.scatter(sim_traj_points['x'], sim_traj_points['y'], color=colors[i], label=f'Sim Traj {i+1}')
    
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('2D Plot of Observations and Simulated Trajectories')
    plt.legend()
    plt.savefig(filename)

 
def main():
    """
    main entry point
    """
    

    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, required=True, help='Path to the map without extensions')
    args = parser.parse_args()
    
    
    work = {'mass': 3.463388126201571, 'lf': 0.15597534362552312, 'tlad': 0.82461887897713965, 'vgain': 0.90338203837889}
    
    with open('config_example_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    planner = PurePursuitPlanner(conf, (0.17145+0.15875)) #FlippyPlanner(speed=0.2, flip_every=1, steer=10)
    
    bm = BicycleModel(planner.wheelbase)
    
    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1, timestep=0.01, integrator=Integrator.RK4)
    #env.add_render_callback(render_callback)
    
    learner = Learner(
        algo = args.algo, 
        rng = 0,
        update_interval = 100,
        num_epochs = 10,
        num_samples = 100
        )

    simulator = Simulator(learner, bm, planner, work)

    # Initialize data storage
    obs_points = {'x': [], 'y': []}
    all_sim_traj_points = []

    obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
    #env.render()
    
     
    laptime = 0.0
    start = time.time()

    step = 0
    while not done and step < 300:
        obs_points['x'].append(obs['poses_x'][0])
        obs_points['y'].append(obs['poses_y'][0])


        step += 1 

        acc, steer = planner.plan(obs, work)
        acc *= 0.5
        noise = [0, 0] #np.random.normal([2]) * 0.001
        
        pred = bm.step(obs, acc + noise[0], steer + noise[1], None, logger)

        learner.collect_data_one_step(obs, acc + noise[0], steer + noise[1],  pred, call_back = collect_data_call_back, logger = logger)



        obs, step_reward, done, info = env.step(np.array([[steer + noise[0], acc + noise[1]]]))
        
        laptime += step_reward
        #env.render(mode='human')
        if False and step > 50 and step % 100 == 1:
            print(f'{step=}') #logger.log(Logging_Level.INFO.value, f'Step {step}')
            learner.update_model(call_back = update_models_call_back, logger = logger)

        if False and step > 200:
            sim_traj = simulator.forward(obs, 200, logger = logger)
            sim_traj_points = {'x': [], 'y': []}
            for sim_obs in sim_traj:
                sim_x = sim_obs['poses_x'][0]
                sim_y = sim_obs['poses_y'][0]
                sim_traj_points['x'].append(sim_x)
                sim_traj_points['y'].append(sim_y)
            all_sim_traj_points.append(sim_traj_points)
        

        step += 1
        
    logger.log(Logging_Level.INFO.value, f'Sim elapsed time: {laptime} | Real elapsed time: {time.time()-start}')
    # Draw the plot after collecting all data
    draw_plot(obs_points, all_sim_traj_points, args.algo + '.png')

if __name__ == '__main__':
    main()  