import itertools
import os
import sys
import gym
import yaml

import einops as ei
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from numba import njit

import random

from typing_extensions import override

from typing import List, Dict, Any

from f110_gym.envs.f110_env import F110Env
from f110_gym.envs.base_classes import Integrator 

from efppo.task.dyn_types import Control, HFloat, LFloat, Obs, State 
from efppo.task.task import Task, TaskState
from efppo.utils.angle_utils import rotx, roty, rotz
from efppo.utils.jax_types import BBFloat, BoolScalar, FloatScalar
from efppo.utils.jax_utils import box_constr_clipmax, box_constr_log1p, merge01, tree_add, tree_inner_product, tree_mac
from efppo.utils.plot_utils import plot_x_bounds, plot_y_bounds, plot_y_goal
from efppo.utils.rng import PRNGKey
from efppo.utils.cfg_utils import RecursiveNamespace


"""
Planner Helpers
"""
@njit(fastmath=False, cache=True)
def nearest_point_on_trajectory(point, trajectory):
    """
    Return the nearest point along the given piecewise linear trajectory.

    Same as nearest_point_on_line_segment, but vectorized. This method is quite fast, time constraints should
    not be an issue so long as trajectories are not insanely long.

        Order of magnitude: trajectory length: 1000 --> 0.0002 second computation (5000fps)

    point: size 2 numpy array
    trajectory: Nx2 matrix of (x,y) trajectory waypoints
        - these must be unique. If they are not unique, a divide by 0 error will destroy the world
    """
    diffs = trajectory[1:,:] - trajectory[:-1,:]
    l2s   = np.sqrt(diffs[:,0]**2 + diffs[:,1]**2)
    # this is equivalent to the elementwise dot product
    # dots = np.sum((point - trajectory[:-1,:]) * diffs[:,:], axis=1)
    dots = np.empty((trajectory.shape[0]-1, ))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t<0.0] = 0.0
    t[t>1.0] = 1.0
    # t = np.clip(dots / l2s, 0.0, 1.0)
    projections = trajectory[:-1,:] + (t*diffs.T).T
    # dists = np.linalg.norm(point - projections, axis=1)
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp*temp))
    min_dist_segment = np.argmin(dists)
    return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment


@njit(fastmath=False, cache=True)
def all_points_on_trajectory_within_circle(point, radius, trajectory, t=0.0, wrap=False):
    """
    Finds all points on the trajectory that are within one radius away from the given point.
    
    Arguments:
    - point: The point to check distances from.
    - radius: The radius within which points on the trajectory are considered "within."
    - trajectory: The Nx2 array of points defining the trajectory.
    - t: The starting time parameter (0 <= t < 1).
    - wrap: Whether to wrap around the trajectory if no intersection is found.
    
    Returns:
    - intersecting_points: A list of all points within the radius.
    - indices: A list of indices in the trajectory corresponding to these points.
    - t_values: A list of t values corresponding to where the points are found on the segments.
    """
    start_i = int(t)
    start_t = t % 1.0
    intersecting_points = []
    indices = []
    t_values = []
    trajectory = np.ascontiguousarray(trajectory)
    
    for i in range(start_i, trajectory.shape[0]-1):
        start = trajectory[i, :]
        end = trajectory[i + 1, :] + 1e-6
        V = np.ascontiguousarray(end - start)

        a = np.dot(V, V)
        b = 2.0 * np.dot(V, start - point)
        c = np.dot(start, start) + np.dot(point, point) - 2.0 * np.dot(start, point) - radius * radius
        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            continue
        
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2.0 * a)
        t2 = (-b + discriminant) / (2.0 * a)

        if i == start_i:
            if t1 >= 0.0 and t1 <= 1.0 and t1 >= start_t:
                intersecting_points.append(start + t1 * V)
                indices.append(i)
                t_values.append(t1)
            if t2 >= 0.0 and t2 <= 1.0 and t2 >= start_t:
                intersecting_points.append(start + t2 * V)
                indices.append(i)
                t_values.append(t2)
        else:
            if t1 >= 0.0 and t1 <= 1.0:
                intersecting_points.append(start + t1 * V)
                indices.append(i)
                t_values.append(t1)
            if t2 >= 0.0 and t2 <= 1.0:
                intersecting_points.append(start + t2 * V)
                indices.append(i)
                t_values.append(t2)

    if wrap:
        for i in range(-1, start_i):
            start = trajectory[i % trajectory.shape[0], :]
            end = trajectory[(i + 1) % trajectory.shape[0], :] + 1e-6
            V = end - start

            a = np.dot(V, V)
            b = 2.0 * np.dot(V, start - point)
            c = np.dot(start, start) + np.dot(point, point) - 2.0 * np.dot(start, point) - radius * radius
            discriminant = b * b - 4 * a * c

            if discriminant < 0:
                continue
            
            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2.0 * a)
            t2 = (-b + discriminant) / (2.0 * a)

            if t1 >= 0.0 and t1 <= 1.0:
                intersecting_points.append(start + t1 * V)
                indices.append(i)
                t_values.append(t1)
            if t2 >= 0.0 and t2 <= 1.0:
                intersecting_points.append(start + t2 * V)
                indices.append(i)
                t_values.append(t2)

    return intersecting_points, indices, t_values

@njit(fastmath=False, cache=True)
def first_point_on_trajectory_intersecting_circle(point, radius, trajectory, t=0.0, wrap=False):
    """
    starts at beginning of trajectory, and find the first point one radius away from the given point along the trajectory.

    Assumes that the first segment passes within a single radius of the point

    http://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm
    """
    start_i = int(t)
    start_t = t % 1.0
    first_t = None
    first_i = None
    first_p = None
    trajectory = np.ascontiguousarray(trajectory)
    for i in range(start_i, trajectory.shape[0]-1):
        start = trajectory[i,:]
        end = trajectory[i+1,:]+1e-6
        V = np.ascontiguousarray(end - start)

        a = np.dot(V,V)
        b = 2.0*np.dot(V, start - point)
        c = np.dot(start, start) + np.dot(point,point) - 2.0*np.dot(start, point) - radius*radius
        discriminant = b*b-4*a*c

        if discriminant < 0:
            continue
        #   print "NO INTERSECTION"
        # else:
        # if discriminant >= 0.0:
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2.0*a)
        t2 = (-b + discriminant) / (2.0*a)
        if i == start_i:
            if t1 >= 0.0 and t1 <= 1.0 and t1 >= start_t:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            if t2 >= 0.0 and t2 <= 1.0 and t2 >= start_t:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break
        elif t1 >= 0.0 and t1 <= 1.0:
            first_t = t1
            first_i = i
            first_p = start + t1 * V
            break
        elif t2 >= 0.0 and t2 <= 1.0:
            first_t = t2
            first_i = i
            first_p = start + t2 * V
            break
    # wrap around to the beginning of the trajectory if no intersection is found1
    if wrap and first_p is None:
        for i in range(-1, start_i):
            start = trajectory[i % trajectory.shape[0],:]
            end = trajectory[(i+1) % trajectory.shape[0],:]+1e-6
            V = end - start

            a = np.dot(V,V)
            b = 2.0*np.dot(V, start - point)
            c = np.dot(start, start) + np.dot(point,point) - 2.0*np.dot(start, point) - radius*radius
            discriminant = b*b-4*a*c

            if discriminant < 0:
                continue
            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2.0*a)
            t2 = (-b + discriminant) / (2.0*a)
            if t1 >= 0.0 and t1 <= 1.0:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            elif t2 >= 0.0 and t2 <= 1.0:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break

    return first_p, first_i, first_t


@njit(fastmath=False, cache=True)
def get_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
    """
    Returns actuation
    """
    waypoint_y = np.dot(np.array([np.sin(-pose_theta), np.cos(-pose_theta)]), lookahead_point[0:2]-position)
    speed = lookahead_point[2]
    if np.abs(waypoint_y) < 1e-6:
        return speed, 0.
    radius = 1/(2.0*waypoint_y/lookahead_distance**2)
    steering_angle = np.arctan(wheelbase/radius)
    return speed, steering_angle

#@njit(fastmath=False, cache=True)
def partition_line(points, n):
    # Ensure that the points array has the right shape
    points = np.asarray(points)
    
    # If all points are the same, create a straight line from start to end
    #if np.all(points == points[0]):
    #    return points.tolist()
    
    # Calculate cumulative distances between consecutive points
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative_distances = np.concatenate(([0], np.cumsum(distances)))
    total_length = cumulative_distances[-1]

    # Uniformly partition the total length into n-1 segments
    partition_lengths = np.linspace(0, total_length, n)

    # Find the boundary points at these partition lengths
    boundary_points = []
    for length in partition_lengths:
        # Find which segment this length falls into
        segment_index = np.searchsorted(cumulative_distances, length) - 1
        segment_index = np.clip(segment_index, 0, len(distances) - 1)
        
        # Calculate the interpolation ratio t
        t = (length - cumulative_distances[segment_index]) / distances[segment_index] if distances[segment_index] != 0 else 0
        
        # Interpolate the point on the segment
        boundary_point = (1 - t) * points[segment_index] + t * points[segment_index + 1] if t != 0 else points[segment_index]
        boundary_points.append(boundary_point)

    return boundary_points



class Planner:
    """
    Example Planner
    """
    def __init__(self, conf, wheelbase):
        
        self.conf = conf
        self.wheelbase = wheelbase 
        #self.load_waypoints(conf)
        self.max_reacquire = 20.

        self.drawn_waypoints = []

        self.cur_waypoint_ids = []
        self.pre_waypoint_ids = []
        """
        loads waypoints
        """
        self.waypoints = np.loadtxt(
            self.conf.wpt_path,
            delimiter=self.conf.wpt_delim, 
            skiprows=self.conf.wpt_rowskip
            )
    
    def load_border(self, conf):
        """
        loads border
        """
        self.border = np.loadtxt(
            os.path.join(conf.base_path, conf.bpt_path), 
            delimiter=conf.bpt_delim, 
            skiprows=conf.bpt_rowskip
            ) #- np.asarray([-52.09937597984384,-48.64305184191518])

    def render_waypoints(self, GL_POINTS, e):
        """
        update waypoints being drawn by EnvRenderer
        """

        #points = self.waypoints

        waypoints = np.vstack((self.waypoints[:, self.conf.wpt_xind], self.waypoints[:, self.conf.wpt_yind])).T
        scaled_points = 50. * waypoints

        for i in range(waypoints.shape[0]):
            if len(self.drawn_waypoints) < waypoints.shape[0]:
                b = e.batch.add(1, GL_POINTS, None, ('v3f/stream', [scaled_points[i, 0], scaled_points[i, 1], 0.]),
                                ('c3B/stream', [183, 193, 222]))
                self.drawn_waypoints.append(b)
            else:
                self.drawn_waypoints[i].vertices = [scaled_points[i, 0], scaled_points[i, 1], 0.]
    
    def render_border(self, GL_POINTS, e):
        border= np.vstack((self.border[:, self.conf.bpt_xind], self.border[:, self.conf.bpt_yind])).T
        
        drawn_border = []
        scaled_points = 50 * border

        for i in range(border.shape[0]):
            if len(drawn_border) < border.shape[0]:
                b = e.batch.add(1, GL_POINTS, None, ('v3f/stream', [scaled_points[i, 0], scaled_points[i, 1], 0.]),
                                ('c3B/stream', [255, 255, 255])) #[183, 193, 222]))
                drawn_border.append(b)
            else:
                drawn_border[i].vertices = [scaled_points[i, 0], scaled_points[i, 1], 0.]
            
            
        
    def _get_current_waypoints(self, lookahead_distance, position, theta):
        """
        gets the current waypoint to follow
        """
        wpts = np.vstack((self.waypoints[:, self.conf.wpt_xind], self.waypoints[:, self.conf.wpt_yind])).T
        nearest_point, nearest_dist, t, i = nearest_point_on_trajectory(position, wpts)
        if nearest_dist < lookahead_distance:
            lookahead_points, i2s, t2s = all_points_on_trajectory_within_circle(position, lookahead_distance, wpts, i+t, wrap=True)
            return np.sort(i2s)
            if i2s == None:
                return None
            current_waypoints = np.empty((len(i2s), 2))
            # x, y
            current_waypoints[:, 0:1] = wpts[i2s, :]
            # speed
            #current_waypoints[:, 2] = 1 #waypoints[i, self.conf.wpt_vind]
            return current_waypoints
        elif nearest_dist < self.max_reacquire:
            return np.append(wpts[i, :], 1).reshape(1, 2) #waypoints[i, self.conf.wpt_vind])
        else:
            return None

    def plan(self, obs, work):
        """
        gives actuation given observation
        """
        pose_x = obs['poses_x'][0]
        pose_y = obs['poses_y'][0]
        pose_theta = obs['poses_theta'][0] 
 
        lookahead_distance = work.tlad
         
        position = np.asarray([pose_x, pose_y])
        self.pre_waypoint_ids = self.cur_waypoint_ids[:]
        self.cur_waypoint_ids = self._get_current_waypoints(lookahead_distance, position, pose_theta)
        waypoints = self.waypoints[self.cur_waypoint_ids]
        lookahead_points = partition_line(waypoints, work.nlad)
 
        return lookahead_points


class F1TenthWayPoint(Task):
    def __init__(self, seed = 10, assets_location = None):
       
        self.dt = 0.05
        self.conf = None
        self.train_map_names = None
        self.test_map_names = None

        self.assets_location = assets_location

        self.load_assets()


        self.cur_map_name = None
        self.cur_mode = 'train'
        self.cur_planner = None
        self.cur_env = None

        self.cur_state = None
        self.cur_collision = None
        self.cur_step = 0

        self.width = 0
 
    def load_assets(self, location = None):
        if self.assets_location is None:
            self.assets_location = os.path.join(
                os.path.dirname(__file__).split('src/efppo/')[0],
                'assets/f110/'
            )
        with open(os.path.join(self.assets_location, 'config.yaml')) as file:
            conf_dict = yaml.load(
                file, 
                Loader=yaml.FullLoader
                )
        self.conf = RecursiveNamespace(**conf_dict)

        map_keys = list(filter(lambda key: 'map' in key, self.conf.sub_spaces))

        if len(map_keys) == 1:
            self.train_map_names = self.test_map_names =  map_keys 
        else:
            random.shuffle(map_keys)
            self.train_map_names = map_keys[:int(np.ceil(len(map_keys) / 2))]
            self.test_map_names = map_keys[len(self.train_map_names):]

    def get_state(self, state_dict: Dict[str, Any], lookahead_points: List[np.ndarray]):
        pose_x = state_dict['poses_x']
        pose_y = state_dict['poses_y']
        
        pose_theta = state_dict['poses_theta']
        velocity_x = state_dict['linear_vels_x']
        velocity_y = state_dict['linear_vels_y']

        assert lookahead_points[0].shape == (2, )

        return np.concatenate((
            pose_x,
            pose_y, 
            pose_theta, 
            velocity_x, 
            velocity_y, 
            *lookahead_points), axis = -1)



    def reset(self, mode: str = 'train'):
        gen_new_map = random.random() > 0.5
        
        if self.cur_mode.lower() != mode.lower() or self.cur_map_name is None:
            gen_new_map = True
                 
        if gen_new_map:
            self.cur_map_name = random.choice(
                self.train_map_names if 'train' in mode.lower() else self.test_map_names
                )
            
            cur_map_conf = getattr(self.conf, self.cur_map_name).copy()
            for attr in cur_map_conf.name_lst:
                if 'path' in attr.lower():
                    setattr(
                        cur_map_conf, 
                        attr, 
                        os.path.join(
                            self.assets_location,
                            getattr(cur_map_conf, attr)
                            )
                    )

            wheelbase = self.conf.work.lf + self.conf.work.lr
            self.cur_planner = Planner(cur_map_conf, wheelbase)

            self.cur_env = gym.make(
                'f110_gym:f110-v0', 
                map=cur_map_conf.map_path,
                map_ext=getattr(self.conf, self.cur_map_name).map_ext, 
                num_agents=1, 
                timestep=self.dt, 
                integrator=Integrator.RK4
                )


        init_pose_ind = np.random.choice(int(self.cur_planner.waypoints.shape[0]/10))
        init_pose = self.cur_planner.waypoints[init_pose_ind].reshape(-1, 2)
        init_pose += np.random.normal(
            loc = 0 * init_pose,  
            scale = 1e-1 * np.ones(init_pose.shape) 
        )
        init_angle = np.random.normal(
            loc = np.zeros(init_pose.shape[:-1]).reshape(-1, 1),
            scale = 3 * np.ones(init_pose.shape[:-1]).reshape(-1, 1)
        )
        init_state = np.concatenate((init_pose, init_angle), axis = -1)
        state, step_reward, done, info = self.cur_env.reset(init_state)
        self.cur_collision = state['collisions']
        self.cur_lookahead_points = self.cur_planner.plan(state, self.conf.work) 
        self.cur_state = self.get_state(state, self.cur_lookahead_points)
        self.cur_step = 0
        return self.cur_state#, 0, done, info
    
    @override
    def get_obs(self, state: State) -> Obs:
        assert state.shape[-1] == 5 + 2 * self.conf.work.nlad
        pose = state[np.array([0, 1])]
        lookahead_points = state[5:].reshape(-1, 2)
        lookahead_fts = (lookahead_points - pose).reshape(-1)
        other_fts = state[2:5]
        return np.concatenate((other_fts, lookahead_fts))
        

    @override
    def step(self, state: State, control: Control) -> State:
        assert control.shape[-1] == 2
        assert state.shape[-1] == 5 + 2 * self.conf.work.nlad
        nxt_state, step_reward, done, info = self.cur_env.step(control.reshape(-1, 2))
        lookahead_points = self.cur_planner.plan(nxt_state, self.conf.work)
        self.cur_state = self.get_state(nxt_state, lookahead_points)
        self.cur_step += 1
        return self.cur_state #, step_reward, done, info
         
    def l(self, state: State, control: Control) -> LFloat:
        weights = np.array([1.2e-2])
        return weights * (self.cur_planner.cur_waypoint_ids[0] - self.cur_planner.pre_waypoint_ids[0])
    
    def h_components(self, state: State) -> HFloat:
        fts = self.get_obs(state)
        offsets = fts[5:].reshape(-1, 2)
        min_dist = np.min(jax.vmap(lambda coord: jnp.sqrt(jnp.dot(coord, coord)))(offsets))
        collision_dist = self.cur_collision * min_dist
        if self.width < 1e-3:
            self.width = collision_dist
        else:
            self.width = min(self.width, collision_dist)
    
        return self.width
    

    def sample_x0_train(self, key: PRNGKey, num: int = 1) -> TaskState:
        state = self.reset()
        return state.reshape(1, *state.shape)

    def should_reset(self, state: State) -> BoolScalar:
        # Reset the state if it is frozen.
        return self.cur_step > 1e6