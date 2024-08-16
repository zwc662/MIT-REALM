import time
import gym
import numpy as np
import yaml
import os

from copy import deepcopy
import argparse
from argparse import Namespace

import jax
#import jax.numpy as jnp
from jax import lax
jax.config.update('jax_platform_name', 'cuda')

from f110_gym.envs.f110_env import F110Env
from f110_gym.envs.base_classes import Integrator 
from sysid.src.planner import PurePursuitPlanner
from sysid.src.model import BicycleModel
from sysid.src.constants import Logging_Level
from sysid.src.learner import Learner
from sysid.src.simulator import Simulator

from sysid.utils.plot import draw_plot

import numpy as np
import math


import logging
logging.basicConfig(
    filename="log",
    filemode='w',
    format='%(name)s -  %(lineno)d - %(message)s')
logger = logging.getLogger()
logger.setLevel(Logging_Level.INFO.value)
logger.log = lambda level, *args, **kwargs: logger.info(*args, **kwargs) if level >= logging.INFO else ''
import wandb
# Use wandb-core
wandb.require("core")
wandb.login()
 
 
class RecursiveNamespace(Namespace):
    def __init__(self, **kwargs):
        self.sub_spaces = list()
        self.name_lst = list()

        dict_kwargs = {}
        name_kwargs = {}
        for name in kwargs:
            if type(kwargs[name]) == dict:
                dict_kwargs[name] = kwargs[name]
                self.sub_spaces.append(name)
            else:
                name_kwargs[name] = kwargs[name]
                self.name_lst.append(name)
        
        super().__init__(**name_kwargs)
        
        for name in dict_kwargs:
            setattr(self, name, RecursiveNamespace(**dict_kwargs[name]))
            self.sub_spaces.append(name)
    
    def update(self, namespace):
        for name in namespace.name_lst:
            setattr(self, name, getattr(namespace, name))

 
 
def collect_data_call_back(self):
    wandb.log({
            'x': self.buffer['Xs']['xs'][-1],
            'y': self.buffer['Xs']['ys'][-1],
            'poses_theta': self.buffer['Xs']['poses_theta'][-1],
            'velocity_x': self.buffer['Xs']['velocity_x'][-1],
            'velocity_y': self.buffer['Xs']['velocity_y'][-1],
            'accelerate': self.buffer['Xs']['accelerate'][-1],
            'steering': self.buffer['Xs']['steering'][-1],
            'pred_x': self.buffer['Ys']['pred_xs'][-1],
            'pred_y': self.buffer['Ys']['pred_ys'][-1],
            'bm_x': self.buffer['Ys']['xs'][-1],
            'bm_y': self.buffer['Ys']['ys'][-1],
            'label_x': self.buffer['Xs']['xs'][-1] - self.buffer['Ys']['xs'][-2] if len(self.buffer['Ys']['xs']) > 1 else 0,
            'label_x': self.buffer['Xs']['ys'][-1] - self.buffer['Ys']['ys'][-2] if len(self.buffer['Ys']['ys']) > 1 else 0,
            'err_x': abs(self.buffer['Xs']['xs'][-1] - self.buffer['Ys']['pred_xs'][-2]) if len(self.buffer['Ys']['pred_xs']) > 1 else 0,
            'err_y': abs(self.buffer['Xs']['ys'][-1] - self.buffer['Ys']['pred_ys'][-2]) if len(self.buffer['Ys']['pred_ys']) > 1 else 0
        }
    )
    

def update_models_call_back(info):
    wandb.log(info)


def run_one_step(env, obs, simulator, work, obs_points, laptime, step):
    obs_points['x'].append(obs['poses_x'][0])
    obs_points['y'].append(obs['poses_y'][0])

    step += 1 
        
    acc, steer = simulator.planner.plan(
        obs, 
        work
        )

    pred_nxt_obs = simulator.bm.step(obs, acc, steer, None, logger)

    simulator.learner.collect_data_one_step(obs, pred_nxt_obs, acc, steer, call_back = collect_data_call_back, logger = logger)

    obs, step_reward, done, info = env.step(np.array([[steer, acc]]))
    
    laptime += step_reward

    return laptime, step, (obs, step_reward, done, info)

    
def run(env, simulator, work, args, logger, train = True):
    waypoints = {
        'x': simulator.planner.waypoints[:, simulator.planner.conf.wpt_xind].tolist(), 
        'y': simulator.planner.waypoints[:, simulator.planner.conf.wpt_yind].tolist()
        }
    
    if args.params == 1:
        # Perturb dynamics
        params = {
            'mu': 0.08985050194246735, 
            'C_Sf': 6.461633476571456, 
            'C_Sr': 5.5608394350107035, 
            'lf': 0.15607724893972566, 
            'lr': 0.24182564159074513, 
            'h': 0.06362648398675183, 
            'm': 4.589965637775949, 
            'I': 0.028435271112269407, 
            's_min': -0.6054291058749389, 
            's_max': 0.057440274511241785, 
            'sv_min': -3.046571558607359, 
            'sv_max': 2.2955848726053194, 
            'v_switch': 10.308033363723174, 
            'a_max': 3.8618681689800463, 
            'v_min': -2.3970177032927924, 
            'v_max': 19.91587002985152, 
            'width': 0.1416204581583914}#, 'length': 0.5438096805680749}
        env.params.update(params)
    elif args.params == 2:
        params = env.params
        for k, v in params.items():
            params[k] = np.random.normal(loc = params[k], scale = 0.5 * abs(params[k]))
    
     
    env.timestep =  simulator.bm.dt
    
    #env.timestep = 0.04
    #env.add_render_callback(render_callback)
   
    # Initialize data storage
    obs_pointss = []
    all_sim_traj_points = []
    
    init_x = simulator.planner.waypoints[0][0]
    init_y = simulator.planner.waypoints[0][1]
     
    obs, step_reward, done, info = env.reset(np.array([[init_x, init_y, 0]]))
     
    last_sim = 0
    sim_interval = 10 

    laptime = 0.0
    start = time.time()
    env.render()
 

    step = 0
    obs_points = {'x': [], 'y': []}
    while step * env.timestep < 6 * sim_interval:

        lap_time, step, (obs, step_reward, done, info) = run_one_step(env, obs, simulator, work, obs_points, laptime, step)
        env.render(mode='human') 
        if step == 3001:
            pass
        if train and step > 50 and step % 100 == 1:
            print(f'Train @ {step=}') 
            logger.log(Logging_Level.INFO.value, f'Train @ {step=}')
            simulator.learner.update_model(call_back = update_models_call_back, logger = logger)
         
        if step * env.timestep >= last_sim + sim_interval:
            print(f'Simulation @ {step=}') 
            logger.info(f'Simulation @ {step=}')
            pre_sim_obs = deepcopy({k: v for k, v in obs.items()})
            sim_traj = simulator.forward(obs, math.floor(sim_interval / simulator.bm.dt), logger = logger)
            for k, v in pre_sim_obs.items():
                assert (np.asarray([v]).flatten() == np.asarray([obs[k]]).flatten()).all()

            sim_traj_points = {'x': [], 'y': []}
            for sim_obs in sim_traj:
                sim_x = sim_obs['poses_x'][0]
                sim_y = sim_obs['poses_y'][0]
                sim_traj_points['x'].append(sim_x)
                sim_traj_points['y'].append(sim_y) 
            all_sim_traj_points.append(sim_traj_points)
             
            obs_pointss.append(obs_points)
            obs_points = {'x': [], 'y': []}
            
            last_sim += step * env.timestep

            try:
                draw_plot(waypoints, obs_pointss, all_sim_traj_points, '_'.join([f'params_{args.params}', f'work_{args.work}', f'algo_{args.algo}', f'{os.path.basename(env.map_name)}.png']))
            except:
                pass

 
    logger.log(Logging_Level.INFO.value, f'Sim elapsed time: {laptime} | Real elapsed time: {time.time()-start}')
    # Draw the plot after collecting all data
    try:
        draw_plot(waypoints, obs_pointss, all_sim_traj_points, '_'.join([f'params_{args.params}', f'work_{args.work}', f'algo_{args.algo}', f'{os.path.basename(env.map_name)}.png']))
    except:
        pass

def main():
    """
    main entry point
    """
   
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, required=True, help='Path to the map without extensions')
    parser.add_argument('--work', type=int, required=False, default = 0, help='Path to the map without extensions')
    parser.add_argument('--params', type=int, required=False, default = 0, help='Path to the map without extensions')
    parser.add_argument('--render', action='store_true', help='render track')
    args = parser.parse_args()
    wandb.init(
        mode="disabled",
        # Set the project where this run will be logged
        project="basic-intro",
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=f"{args.algo}_forward_sim",
        # Track hyperparameters and run metadata
        )
    with open('config_example_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = RecursiveNamespace(**conf_dict)
    
    work = {'mass': 3.463388126201571, 'lr': 0.17145, 'lf': 0.15597534362552312, 'tlad': 0.82461887897713965, 'vgain': 0.90338203837889}
    if args.work == 1:
        # perturb planner
        work = {'mass': 7.319960348652749, 'lr': 0.2543618539794769, 'lf': 0.3438519625296129, 'tlad': 0.5460865538470376, 'vgain': 2.684377437220065}

    #work['tlad'] = 4.0
    #work['vgain'] = 1
    #work['mass'] = .1
    #work['lr'] *= 5

    
    wb =  work['lf'] + work['lr']
    
    planner = PurePursuitPlanner(conf, wb) #FlippyPlanner(speed=0.2, flip_every=1, steer=10)
    
    bm = BicycleModel(
        lf = work['lf'], 
        lr = work['lr'],
        d =  0., #.1 - work['vgain'],
        m = work['mass'],
        dt = 0.01)
    
     
    learner = Learner(
        algo = args.algo, 
        rng = 0,
        update_interval = 100,
        num_epochs = 10,
        num_samples = 100
        )
    

    simulator = Simulator(learner, bm, planner, work)

    
   

    #np.random.shuffle(conf.sub_spaces)
    for name in conf.sub_spaces[1:2]:
        
            
        simulator.planner.conf.update(getattr(simulator.planner.conf, name))
        simulator.planner.load_waypoints(getattr(simulator.planner.conf, name))
        simulator.planner.load_border(getattr(simulator.planner.conf, name)) 

        env = gym.make('f110_gym:f110-v0', map=simulator.planner.conf.map_path, map_ext=simulator.planner.conf.map_ext , num_agents=1, timestep=0.01, integrator=Integrator.RK4)

        print(f"{name} in {simulator.planner.conf.map_path + simulator.planner.conf.map_ext}")
        
        if args.render:
            from pyglet.gl import GL_POINTS
            def render_callback(env_renderer):
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
 
                simulator.planner.render_waypoints(GL_POINTS, e)
                simulator.planner.render_border(GL_POINTS, e)
     
            env.render_callbacks.append(render_callback)
            
        else:
            env.render = lambda **kwargs: None
         
        run(env, simulator, work, args, logger, train = True if 'train' in name else False)

        env.renderer = None
        env = None
        GL_POINTS = None 
        


if __name__ == '__main__':
    main()  