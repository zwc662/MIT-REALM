import itertools
import functools as ft
import os
import sys
import gym
import yaml

import einops as ei
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
from jax import lax
 

import numpy as np
from numba import njit

import random

import warnings

from typing_extensions import override

from typing import List, Dict, Any, Union, Optional

from f110_gym.envs.f110_env import F110Env
from f110_gym.envs.base_classes import Integrator 
from f110_gym.envs.collision_models import get_vertices

from matplotlib import pyplot as plt

from efppo.task.dyn_types import Control, HFloat, LFloat, Obs, State, Action
from efppo.task.task import Task, TaskState
from efppo.utils.angle_utils import rotx, roty, rotz
from efppo.utils.jax_types import BBFloat, BoolScalar, FloatScalar
from efppo.utils.jax_utils import box_constr_clipmax, box_constr_log1p, merge01, tree_add, tree_inner_product, tree_mac, plain_cond
from efppo.utils.plot_utils import plot_x_bounds, plot_y_bounds, plot_y_goal
from efppo.utils.rng import PRNGKey
from efppo.utils.cfg_utils import RecursiveNamespace
from efppo.utils.rng import PRNGKey
from efppo.utils.tfp import tfd
 
"""
Planner Helpers
"""
#@njit(fastmath=False, cache=True)
@jax.jit
def _jit_nearest_point_on_trajectory_(point, trajectory):
    """
    Return the nearest point along the given piecewise linear trajectory.

    Same as nearest_point_on_line_segment, but vectorized. This method is quite fast, time constraints should
    not be an issue so long as trajectories are not insanely long.

        Order of magnitude: trajectory length: 1000 --> 0.0002 second computation (5000fps)

    point: size 2 numpy array
    trajectory: Nx2 matrix of (x,y) trajectory waypoints
        - these must be unique. If they are not unique, a divide by 0 error will destroy the world


    >>>>>>>> If allow wrapping the track, when the vehicle approaches the end point, a lower indexed waypoint in front of the vehicle must be treated as the lookahead point

    """
    diffs = trajectory[1:,:] - trajectory[:-1,:]
    diffs = jnp.concatenate((diffs, (trajectory[0] - trajectory[-1]).reshape(1, -1)), axis = 0)
    l2s   = jnp.sqrt(jnp.square(diffs).sum(axis = -1))
    # this is equivalent to the elementwise dot product
    # dots = np.sum((point - trajectory[:-1,:]) * diffs[:,:], axis=1)
    dots = jax.vmap(lambda wp, diff: jnp.dot((point - wp), diff))(trajectory, diffs)
    t = dots / l2s
    t = jnp.where(t<0.0, 0, t)
    t = jnp.where(t>1.0,1.0, t) #.at[t>1.0].set(1.0)
    # t = np.clip(dots / l2s, 0.0, 1.0)
    projections = trajectory + (t*diffs.T).T
    # dists = np.linalg.norm(point - projections, axis=1)
    dists = jax.vmap(lambda proj_p: jnp.sqrt(np.sum(jnp.square(point - proj_p))))(projections)
    min_dist_segment = jnp.argmin(dists)
    return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment


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
    diffs = np.concatenate((diffs, (trajectory[0] - trajectory[-1]).reshape(1, -1)), axis = 0)
    l2s   = np.sqrt(np.square(diffs).sum(axis = -1))
    # this is equivalent to the elementwise dot product
    # dots = np.sum((point - trajectory[:-1,:]) * diffs[:,:], axis=1)
    dots = np.empty((trajectory.shape[0], ))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t<0.0] = 0.0
    t[t>1.0] = 1.0
    # t = np.clip(dots / l2s, 0.0, 1.0)
    projections = trajectory + (t*diffs.T).T
    # dists = np.linalg.norm(point - projections, axis=1)
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp*temp))
    min_dist_segment = np.argmin(dists)
    return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment



#@njit(fastmath=False, cache=True)
#@ft.partial(jax.jit, static_argnames = ['radius', 't', 'wrap', 'trajectory_length'])
def all_points_on_trajectory_within_circle_confusing_version(point, radius, trajectory, t=0.0, wrap=False):
    """
    Finds all points on the trajectory that are within one radius away from the given point.

    Between each consecutive (`start`, `end`) points, knowing that `start` is within radius
        * Check if `end` is also inside radius by verifying whether there exists a `t \in [0, 1]` such that 
                `(point, start, t * start + (1 - t) * end)`  forms a triangle
        * If `(||point, start|| + ||start, t * start + (1 - t) * end||)^2 <= ||radius||^2`, 
            then `||point, t * start + (1 - t) * end|| < ||radius||` due to triangle law:
                `||point, t * start + (1 - t) * end|| < ||point, start|| + ||start, t * start + (1 - t) * end||`
        * Solving `(||point, start|| + ||start, t * start + (1 - t) * end||)^2 <= ||radius||^2` by checking if
            ((start - point) + (start - end) * t)**2 - radius**2 == 0 has real solution
            ==> (start - end)**2 * t**2 + 2 * (start - point) * (start - end) * t + (start - point)**2 - radius**2 == 0
                |------- a ----|         |----------------- b ---------------|     |---------------- c ----------|
    
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
    start_i = jnp.asarray(t).astype(int)
    start_t = t % 1.0
    end_i = start_i + 1
    trajectory = jnp.asarray(trajectory)
    tot_i = trajectory.shape[0] 

    def continue_search(idx):
        i = jnp.mod(idx, tot_i).astype(int)
        i_nxt = jnp.mod(idx + 1, tot_i).astype(int)
         
         
        start = trajectory[i]
        end = trajectory[i_nxt] + 1e-6

        V = jnp.asarray(end - start)
        U = jnp.asarray(point - start)
        a = jnp.dot(V, V)
        b = 2.0 * jnp.dot(V, U)
        c = jnp.dot(U, U) - radius * radius

     
        #c = np.dot(start, start) + np.dot(point, point) - 2.0 * np.dot(start, point) - radius * radius
         
        discriminant = b * b - 4 * a * c

        def helper(discriminant):
            discriminant = jnp.sqrt(discriminant) 
            t1 = (-b - discriminant) / (2.0 * a)
            t2 = (-b + discriminant) / (2.0 * a)
 
            ret = jnp.all(
                jnp.logical_or(
                    jnp.logical_and(
                        jnp.logical_and(t1 >= 0.0, t1 <= 1.0),
                        jnp.logical_or(i != start_i, t1 >= start_t)
                        ),
                        jnp.logical_and(
                            jnp.logical_and(t2 >= 0.0, t2 <= 1.0),
                            jnp.logical_or(i != start_i, t2 >= start_t)
                        )
                )
            )
            return ret
        
        ret = jnp.all(jnp.logical_or(discriminant < 0, helper (discriminant)))    

        return ret
    
    end_i = int(lax.while_loop(
        continue_search, 
        lambda i: jnp.mod(i + 1, tot_i).astype(int),
        end_i
        ))
    
    indices = jax.jit(
        lambda last_t: lax.cond(jnp.all(last_t <= t), 
                                jnp.concatenate(jnp.arange(t, trajectory.shape[0]), jnp.arange(last_t)),
                                jnp.arange(t, last_t)
        ), static_argnums=[0,]
    )(end_i)
         
    return indices
 
def all_points_on_trajectory_within_circle(point, radius, trajectory, start_i):
    """
    Finds all points on the trajectory that are within one radius away from the given point.

    Arguments:
    - point: The point to check distances from.
    - radius: The radius within which points on the trajectory are considered "within."
    - trajectory: The Nx2 array of points defining the trajectory.
    - start_i: The starting index on the trajectory.
    
    Returns:
    - indices: A list of indices in the trajectory corresponding to the points within the radius.
    """
    start_i = int(start_i % len(trajectory))
    
    # Compute squared distances to avoid square root calculation
    distances_squared = jnp.sum((trajectory - point) ** 2, axis=-1)
    radius_squared = radius ** 2

     
    # Initialize empty arrays to accumulate indices
    suffix_end = lax.cond(
        pred = jnp.logical_and(start_i + 1 < len(trajectory), distances_squared[start_i + 1] < radius_squared),
        true_fun = lambda: jnp.array(start_i + 1),
        false_fun = lambda: jnp.array(start_i, dtype=jnp.int32)
    )
    suffix_end = lax.while_loop(
        cond_fun = lambda suffix_end: jnp.logical_and(
            jnp.logical_and(suffix_end > start_i, suffix_end + 1 < len(trajectory)), 
            jnp.all(distances_squared[suffix_end + 1] < radius_squared)
        ), 
        body_fun = lambda suffix_end: suffix_end + 1, 
        init_val = suffix_end
        )


    # Create prefix only if suffix ends at the last waypoints
    prefix_end = lax.cond(
        pred = jnp.logical_and(
            suffix_end == len(trajectory) - 1, 
            jnp.logical_and(start_i > 0, distances_squared[0] < radius_squared)
            ),
        true_fun = lambda: jnp.array(0),
        false_fun = lambda: jnp.array(-1, dtype=jnp.int32)
    )
    prefix_end = lax.while_loop(
        cond_fun = lambda prefix_end: jnp.logical_and(
            jnp.logical_and(prefix_end >= 0, prefix_end + 1 < start_i), 
            jnp.all(distances_squared[prefix_end + 1] < radius_squared)
        ), 
        body_fun = lambda prefix_end: prefix_end + 1, 
        init_val = prefix_end
        )
    
    # Combine start_i, prefix, and prefix to form final indices list
    combined_indices = np.concatenate((
        np.array([int(start_i)]), 
        np.arange(start_i + 1, np.asarray(suffix_end + 1).item()), 
        np.arange(np.asarray(prefix_end + 1).item())
    ))
    
    return combined_indices
    



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

 
@ft.partial(jax.jit, static_argnames=['n'])
def partition_line(points, n):
    # Calculate cumulative distances between consecutive points
    #print(f"Points shape: {points.shape}. Check previous condition: {jnp.all(points == points[0])}") 
    points = jnp.asarray(points)
    square_distances = jnp.square(points[1:] - points[:-1]).sum(axis=-1)
    cumulative_square_distances = jnp.concatenate((jnp.zeros([1]), jnp.cumsum(square_distances)))
    total_square_length = cumulative_square_distances[-1]

    # Uniformly partition the total length into n-1 segments
    partition_square_lengths = jnp.linspace(0, total_square_length, n)
    
    # Find the boundary points at these partition lengths
    
    #for partition_length in partition_lengths:
    # Find the boundary points at these partition lengths
    segment_indices = jnp.searchsorted(cumulative_square_distances, partition_square_lengths[1:-1], side='right')
    #assert jnp.all(segment_indices > 0)             
    # Calculate the interpolation ratio t
    # segment_index must be greater than zero because cumulative_suqare_distances[0] = 0
    ts = (cumulative_square_distances[segment_indices] - partition_square_lengths[1:-1]) / (cumulative_square_distances[segment_indices] - cumulative_square_distances[segment_indices-1])
    #assert jnp.all(ts > 0)  
    # Interpolate the point on the segment
    boundary_points = points[segment_indices - 1] * ts[:, jnp.newaxis] + points[segment_indices] * (1 - ts)[:, jnp.newaxis]
    
    return jnp.concatenate((jnp.asarray(points[0].reshape(1, -1)), boundary_points, jnp.asarray(points[-1]).reshape(1, -1)))
    
@njit(fastmath=False, cache=True)
def get_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
    """
    Returns actuation
    """
    waypoint_y = np.dot(np.array([np.sin(-pose_theta), np.cos(-pose_theta)]), lookahead_point[0:2]-position)
    speed = 1 #lookahead_point[2]
    if np.abs(waypoint_y) < 1e-6:
        return speed, 0.
    radius = 1/(2.0*waypoint_y/lookahead_distance**2)
    steering_angle = np.arctan(wheelbase/radius)
    return speed, steering_angle


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
        scaled_points = 50 * waypoints

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
        #nearest_point, nearest_dist, t, i = nearest_point_on_trajectory(position, wpts)
        nearest_point, nearest_dist, t, i = _jit_nearest_point_on_trajectory_(position, wpts)
        if nearest_dist < lookahead_distance:
            start_i = np.ceil(i+t).astype(np.int32).item()  % len(wpts)
            i2s = all_points_on_trajectory_within_circle(position, lookahead_distance, wpts, start_i)
            return np.asarray(i2s) #np.sort(i2s)
        else:
            return np.asarray([i])
        '''
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
        '''

    def plan(self, state, work):
        """
        gives actuation given observation
        """
        pose_x = state['poses_x'][0]
        pose_y = state['poses_y'][0]
        pose_theta = state['poses_theta'][0] % (2 * np.pi)
 
        lookahead_distance = work.tlad
         
        position = np.asarray([pose_x, pose_y])
        waypoint_ids = self._get_current_waypoints(lookahead_distance, position, pose_theta)
        waypoints = self.waypoints[waypoint_ids]
        # If all points are the same, create a straight line from start to end
        
        if len(waypoint_ids) <= 2:
            lookahead_points = jnp.linspace(waypoints[0], waypoints[-1], work.nlad)
        else:
            lookahead_points = partition_line(points = waypoints, n = work.nlad) 

        speed, steer = get_actuation(pose_theta, waypoints[-1], position, lookahead_distance, self.wheelbase)
        speed = work.vgain * speed

 
        return lookahead_points, waypoint_ids, np.asarray([[steer, speed]])



class F1TenthWayPoint(Task):
    STATE_X = 0
    STATE_Y = 1
    STATE_YAW = 2
    STATE_VEL_X = 3
    STATE_VEL_Y = 4
    STATE_FST_LAD = 5

    OBS_YAW = 0
    OBS_VEL_X = 1
    OBS_VEL_Y = 2
    OBS_FST_LAD = 3

    PLOT_2D_INDXS = [STATE_X, STATE_Y]


    def __init__(self, seed = 10, assets_location = None, control_mode = ''):
       
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
        self.pre_state = None
        self.cur_state = None
        self.cur_action = None
        self.cur_pursuit_control = None
        self.cur_pursuit_action = None
        self.cur_state_dict = {}
        self.cur_collision = np.asarray([0])
        self.cur_overflow = np.asarray([0])
        self.cur_done = np.asarray([1.])
        self.cur_step = 0

        self.cur_totl = 0

        self.cur_waypoint_ids = None
        self.pre_waypoints_ids = None
 
        self._lb = np.array([-np.pi/2., 3])
        self._ub = np.array([np.pi/2., 3])
        
        self.render = False

        if control_mode not in ['pursuit', '']:
            raise NotImplementedError
        self.control_mode = control_mode

        
        def get_discrete_actions():
            return [[steer, self._ub[1]] for steer in np.linspace(self._lb[0], self._ub[0], 20)]
            '''
            controls = list(zip(self._lb, self._ub)) 
            controls = np.stack(list(itertools.product(*controls)), axis=0)
            controls = np.concatenate([np.zeros((1, 2)), controls], axis=0)
            return controls
            '''
        
        self.discrete_actions = get_discrete_actions()
        

    @property
    def nx(self):
        return 5 + 2 * self.conf.work.nlad

    @property
    def nobs(self):
        return 3 + 2 * self.conf.work.nlad
    
    @property
    def nu(self):
        return 2
        
    @property
    def lb(self):
        return self._lb
    
    @property
    def ub(self):
        return self._ub
     
    @property
    def x_labels(self) -> list[str]:
        base = [
            r"$x$",
            r"$y",
            r"$\theta$",
            r"$v_x$",
            r"$v_y$"
        ] 

        for i in range(self.conf.work.nlad):
            base += [f"lad_{i}_x", f"lad_{i}_y"]
        assert len(base) == self.nx
        return base
     
 
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
            self.test_map_names = map_keys[min(len(map_keys) - 1, len(self.train_map_names)):]

    def pose_from_random_waypoint(self):
        init_pose_ind = np.random.choice(int(self.cur_planner.waypoints.shape[0]))
        return self.pose_from_waypoint(init_pose_ind)
    
    def pose_from_nearest_waypoint(self):
        init_pose_ind = 0
        return self.pose_from_waypoint(init_pose_ind)

    def pose_from_waypoint(self, init_pose_ind: int):
        init_pose = self.cur_planner.waypoints[init_pose_ind][np.array(
            [self.cur_planner.conf.wpt_xind, self.cur_planner.conf.wpt_yind]
            )]
        
        init_angle = np.random.normal(
            loc = np.zeros([1]),
            scale = 0.01 * np.pi
        )
        nxt_pose = self.cur_planner.waypoints[(init_pose_ind + 1) % len(self.cur_planner.waypoints)][np.array(
            [self.cur_planner.conf.wpt_xind, self.cur_planner.conf.wpt_yind]
            )]
        pose_diff = (nxt_pose-init_pose)[np.array(
            [self.cur_planner.conf.wpt_yind, self.cur_planner.conf.wpt_xind]
            )]
        init_angle += np.arctan2(*pose_diff).reshape(1)
        init_pose = np.concatenate((init_pose, init_angle))
        return init_pose


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
 
   
    def pre_reset(self, mode: str = 'train', random_map = False):
        if 'render' in mode.lower():
            self.render = True

        self.cur_state = None
        self.cur_action = None
        self.cur_state_dict = {}
        
        self.cur_collision = np.asarray([0])
        self.cur_overflow = np.asarray([0])
        self.cur_done = np.asarray([-1])

        self.cur_step = 0

        self.cur_totl = 0

         
        self.cur_waypoint_ids = None
        self.pre_waypoints_ids = None

 
        if (random_map or self.cur_planner is None or self.cur_env is None):
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
            self.cur_env = gym.make(
                'f110_gym:f110-v0', 
                map=cur_map_conf.map_path,
                map_ext=getattr(self.conf, self.cur_map_name).map_ext, 
                num_agents=1, 
                timestep=self.dt, 
                integrator=Integrator.RK4
                )
             
            self.cur_planner = Planner(cur_map_conf, wheelbase = self.conf.work.lf + self.conf.work.lr)
 
            self.cur_env.timestep =  self.dt

        self.cur_env.renderer = None
   
    def reset(self, mode: str = 'train', random_map = False, init_pose = None):
        self.pre_reset(mode, random_map)
        
        if init_pose is None:
            init_pose = self.pose_from_random_waypoint()
            if 'soft' in mode.lower():
                init_pose = self.pose_from_nearest_waypoint() 
                 
        print(f"Reset from {init_pose}")
        state_dict, step_reward, done, info = self.cur_env.reset(init_pose.reshape(1, -1))
        
        cur_state = self.post_reset(state_dict)

        return cur_state

    def post_reset(self, state_dict: Dict[str, Any]):
        self.cur_collision = state_dict['collisions']
        self.cur_state_dict = {k: v for k, v in state_dict.items()}
        lookahead_points, waypoint_ids, pursuit_control = self.cur_planner.plan(state_dict, self.conf.work) 
        
        self.cur_pursuit_control = pursuit_control
        self.cur_pursuit_action = self.cts_to_discr(self.cur_pursuit_control)

        self.pre_waypoint_ids = waypoint_ids[:]
        self.cur_waypoint_ids = waypoint_ids[:]
        if not self.check_lad(lookahead_points):
            lookahead_points = self.cur_lookahead_points[:]
        else: 
            self.cur_lookahead_points = lookahead_points[:]

        self.cur_state = self.get_state(state_dict, lookahead_points)
        self.cur_step = 0
 
        if self.render:  
            input("render and reset. Say something ????")  
            self.init_render_callbacks()
            self.cur_env.render(mode='human')
             
             
        return self.cur_state #, 0, done, info
 

    
    def get_lookahead_vertices(self):
        scaled_points = self.cur_lookahead_points
        diffs = scaled_points[1:] - scaled_points[:-1]
        #print(diffs.shape)
        angles = np.arctan2(diffs[:, self.cur_planner.conf.wpt_yind], diffs[:, self.cur_planner.conf.wpt_xind]).flatten().tolist()
        angles.append(angles[-1])
        angles = np.asarray(angles).reshape(-1, 1)
        scaled_points = np.concatenate((scaled_points, angles), axis = -1)
        return scaled_points
        
    def render_lookahead_points(self, GL_QUADS, e, CAR_LENGTH, CAR_WIDTH): 
        unscaled_points = self.get_lookahead_vertices()
        
        if not hasattr(e, "lookahead_points"):
            e.lookahead_points = []
            for i in range(unscaled_points.shape[0]):
                vertices_np = 50 * get_vertices(np.array(unscaled_points[i]), CAR_LENGTH / 2.0, CAR_WIDTH  / 2.0)
                vertices = list(vertices_np.flatten())
                e.lookahead_points.append(
                    e.batch.add(
                        4, 
                        GL_QUADS, 
                        None, 
                        ('v2f', vertices), 
                        ('c3B', (np.asarray([172, 97, 185, 172, 97, 185, 172, 97, 185, 172, 97, 185]) - (i + 1) * unscaled_points.shape[0]).tolist())
                    )
                )
        else: 
            for i in range(unscaled_points.shape[0]):
                vertices_np = 50 * get_vertices(np.array(unscaled_points[i]), CAR_LENGTH / 2.0, CAR_WIDTH  / 2.0)
                vertices = list(vertices_np.flatten())
                e.lookahead_points[i].vertices = vertices 
        
 

    def init_render_callbacks(self):
        from f110_gym.envs.rendering import CAR_LENGTH, CAR_WIDTH
        from pyglet.gl import GL_POINTS, GL_QUADS
        ### For all waypoints along the centerline
        def render_callback(env_renderer):
            # custom extra drawing function

            e = env_renderer

            # update camera to follow car5
            x = e.cars[0].vertices[::2]
            y = e.cars[0].vertices[1::2]
            top, bottom, left, right = max(y), min(y), min(x), max(x)
            e.score_label.x = left
            e.score_label.y = top - 700
            e.left = left - 800
            e.right = right + 800
            e.top = top + 800
            e.bottom = bottom - 800

            self.cur_planner.render_waypoints(GL_POINTS, e)
            self.render_lookahead_points(GL_QUADS, e, CAR_LENGTH, CAR_WIDTH)

        self.cur_env.render_callbacks.append(render_callback)
        return render_callback
      
     

    def check_lad(self, lookahead_points):
        for lookahead_point in lookahead_points:
            if np.isnan(lookahead_point).any() or np.isinf(lookahead_point).any():
                return False
        return True

    @override
    def get_obs(self, state: State) -> Obs:

        assert state.shape[-1] == self.nx, f"{state.shape=}"
         
        pose = state[np.array([self.STATE_X, self.STATE_Y])]
        lookahead_points = state[self.STATE_FST_LAD:].reshape(-1, 2)
        lookahead_fts = (lookahead_points - pose).reshape(-1)
        other_fts = state[self.STATE_Y + 1:self.STATE_FST_LAD]
        return np.concatenate((other_fts, lookahead_fts))
    
    def efppo_control_transform(self, control):
        ## For Continuous Control
        ### The control policy's output mean is constrained to be within (0, 3) using sigmoid scaling (check original /efppo/src/efppo/networks/poly_net.py)
        ### Therefore, the input control needs to be linearly transformed to be within [self.lb, self.ub]
        ### return np.asarray(control.reshape(2) * (self.ub - self.lb) / 3. + self.lb).reshape(-1, 2)

        ## For Discrete Control
        ### The control policy's output mean is constrained to be either 0 or 1 using Categorical distribution (check current /efppo/src/efppo/networks/poly_net.py)
        ### Therefore, the input control needs to be linearly transformed to be within [self.lb, self.ub]
        return self.discr_to_cts(control) #np.asarray((control).reshape(2) * (self.ub - self.lb) / 2. + self.lb).reshape(-1, 2)


    @override
    def step(self, state: State, control: Union[Control, Action]) -> State:
        ## Ensure pausing the simulator when the agent is already out of bound (out of lane boundary or inf/nan in states)
        if False and np.any(self.cur_done > 0.): 
            print(f'Simulation fronzen @ {self.cur_step}: {self.cur_state_dict}')
            if self.render:
                input(f'Simulation fronzen @ {self.cur_step}: {self.cur_state_dict}. Say something ??')
            return self.cur_state
        
       

        #assert control.shape[-1] == 2 or control.shape == (2,), f"{control}"
        #assert control.shape[-1] == 1
        assert state.shape[-1] == self.nx

        if np.asarray([control]).flatten().shape[0] > 1:
            self.cur__action = np.asarray([control]).flatten().reshape(1, self.nu)
        else:
            self.cur__action = self.discr_to_cts(control)(control)

        #action = np.clip(control.reshape(-1, 2), self.lb, self.ub)
       
        self.cur_action = getattr(self, f"cur_{self.control_mode}_action")

        #print(f'Before projection {self.cur_action=}')
        self.cur_action = self.discr_to_cts(self.cts_to_discr(self.cur_action))
        #print(f'After projection {self.cur_action=}')

        if 'pursuit' in self.control_mode and self.render:
            print(f'{self.expert_action=}')
            input('Enter to proceed')

        nxt_state_dict, step_reward, done, info = self.cur_env.step(self.cur_action)
        #print(self.cur_step, nxt_state_dict, action)
             
        if self.render:
            #print(f"control: {control}")
            #print(f"state: {nxt_state_dict}")
            #self.init_render_callbacks()
            self.cur_env.render(mode='human')
          
        
        
        self.cur_collision = nxt_state_dict['collisions']

        nxt_state = np.empty(self.nx)
        #if np.all(self.cur_collision <= 0):
        nxt_lookahead_points, nxt_waypoint_ids, nxt_pursuit_control = self.cur_planner.plan(nxt_state_dict, self.conf.work) 
        
        self.cur_pursuit_control = nxt_pursuit_control
        self.cur_pursuit_action = self.cts_to_discr(self.cur_pursuit_control)

        nxt_state = self.get_state(nxt_state_dict, nxt_lookahead_points)
            
        #print(f"Step: {self.cur_step} | Current pos: {self.cur_state[self.get2d_idxs()]} | Current action: {self.cur_action} | Target waypoints ids: {waypoint_ids} | Target waypoints: {self.cur_planner.waypoints[waypoint_ids]} | Lookahead points: {lookahead_points}")
        #else:
        
        if np.any(self.cur_collision > 0) and self.render:
                print(f'Collision @ {self.cur_step}: state {self.cur_state}')
                #input(f'Collision @ {self.cur_step}: state {self.cur_state}')
                #pass
 
        if np.any(np.isnan(nxt_state)) or \
            np.any(np.isinf(nxt_state)) or \
                np.any(np.abs(nxt_state) > 1e6):
            if self.render:
                print(f"State overflow: {nxt_state} @ {self.cur_step}. Frozen state: {self.cur_state}")
                input(f"State overflow: {nxt_state} @ {self.cur_step}. Frozen state: {self.cur_state}. Say something.")
            self.cur_overflow = np.array([1])
        
        if np.any(self.cur_overflow > 0.): 
            if self.render:
                input(f'Simulation fronzen @ {self.cur_step}: {self.cur_state_dict}')
                F110Env.renderer = None
            self.cur_done = np.asarray([1])
        else:
            assert (nxt_state[self.get2d_idxs()] == (nxt_state_dict['poses_x'][0], nxt_state_dict['poses_y'][0])).all(), f"State dict: {(nxt_state_dict['poses_x'][0], nxt_state_dict['poses_y'][0])} vs nxt_state: {nxt_state[self.get2d_idxs()]}"
            
            self.pre_waypoint_ids = self.cur_waypoint_ids[:] if self.cur_waypoint_ids is not None else nxt_waypoint_ids[:]
            self.cur_waypoint_ids = nxt_waypoint_ids[:]

            if not self.check_lad(nxt_lookahead_points):
                nxt_lookahead_points = self.cur_lookahead_points[:]
            self.cur_lookahead_points = nxt_lookahead_points[:]
            
            self.cur_state_dict = {k: v for k, v in nxt_state_dict.items()}
            self.pre_state = self.cur_state
            self.cur_state = nxt_state
            self.cur_step += 1
        return self.cur_state #, step_reward, done, info
    
    def get_expert_control(self, *args, **kwargs) -> Control:
        return self.cur_pursuit_control
     
    def get_expert_action(self, *args, **kwargs) -> Action:
        return self.cur_pursuit_action
         
    def l1(self, state: State, control: Control) -> LFloat:
        weights = 1 #np.array([1.2e-2])
        return (weights * (self.cur_waypoint_ids[0] - self.pre_waypoint_ids[0] - 1)).item() + \
            self.cur_collision.item() + \
            np.sqrt(np.sum(state[self.STATE_FST_LAD:]**2)).item()


    def l(self, state: State, control: Union[Control, Action, tfd.Distribution]) -> LFloat:
        # Initalize
        l = 0

        ## High velocity => low cost
        if False:
            l = - np.square(state[np.asarray([self.STATE_VEL_X, self.STATE_VEL_Y])]).sum()

        
        ## Stability: greater dist to previous lookahead dist => high cost
        if self.pre_waypoint_ids is not None and self.pre_state is not None:
            previous_lookahead_point = np.asarray(self.cur_planner.waypoints[self.pre_waypoint_ids[-1]])
            l += np.square(np.asarray(self.get2d(state)).reshape(2) - previous_lookahead_point.reshape(2)).sum() - \
                np.square(np.asarray(self.get2d(self.pre_state)).reshape(2) - previous_lookahead_point.reshape(2)).sum()
            
        ## Compare agent control w/ expert control
        if False:
            if hasattr(control, logprob):
                if hasattr(control, 'logits'):
                    l -= control.logprob(self.get_expert_action())
                elif hasattr(control, 'loc'):
                    l -= control.logprob(self.get_expert_control())
                else:
                    raise NotImplementedError
            elif np.asarray([control]).flatten().shape[0] == 1:
                l -= int(np.any(control == self.get_expert_action()))
            elif type(np.asarray([control]).flatten()[0]) == float:
                l -= np.abs(control - self.get_expert_control())
            else:
                raise NotImplementedError

         
        self.cur_totl += l
        
        ## Avoidance: stay close to the nearest lookahead point
        
        if self.cur_waypoint_ids is not None:
            ## Use deviation from nearest waypoint as cost
            nearest_lookahead_point = np.asarray(self.cur_planner.waypoints[self.cur_waypoint_ids[0]])
            l += np.square(np.asarray(self.get2d(state)).reshape(2) - nearest_lookahead_point.reshape(2)).sum()
        
        if np.any(self.cur_collision > 0):
            ## Guaranteed overwhelmed cost for collision
            l = np.abs(self.cur_totl)
    
        return l
            
    
    def h_components(self, state: State) -> HFloat:
        return (np.stack((self.cur_collision, self.cur_overflow)) * 2 - 1.).reshape(len(self.h_labels))

        #return  - np.ones(len(self.h_labels)) # (np.stack((self.cur_collision, self.cur_overflow)) * 2 - 1.).reshape(len(self.h_labels))
    
        return self.cur_collision.item() + self.cur_done.item()
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
        return jnp.asarray(np.random.normal(loc = np.zeros([num, self.nx]), scale = 0.7 * 1e-2 * np.ones([num, self.nx])))

    def should_reset(self, state: Optional[State] = None) -> BoolScalar:
        # Reset the state if it is frozen.
        return np.any(np.logical_or(self.cur_done > 0, self.cur_collision > 0))


        # Reset the state only if it is overflowing
        #return np.any(self.cur_done > 0)

    def get_x0_eval(self) -> TaskState:
        state = self.reset(mode='test')
        return state.reshape(1, *state.shape)
    
    def discr_to_cts(self, control_idx) -> Control:
        """Convert discrete control to continuous."""
        return np.array(self.discrete_actions)[control_idx].reshape(1, self.nu)
    
    def cts_to_discr(self, control: Control) -> int:
        flat_control = np.array(control).reshape(-1)
 
        min_diff = (0, np.abs(flat_control).sum())
        for i in range(1, self.n_actions):
            diff = np.abs(np.asarray(self.discrete_actions[i]).reshape(-1) - flat_control).sum()
            if diff <= min_diff[1]:
                min_diff = (i, diff)
        #assert np.any(np.asarray(self.discrete_actions).reshape(-1, self.nu) == np.asarray(proj_control).reshape(-1)), f'{control} projected as {proj_control} not found in {self.discrete_actions}'
        return min_diff[0]

    @property
    def n_actions(self) -> int:
        return len(self.discrete_actions)
    
    @property
    def h_labels(self) -> int:
        return ["pose", "theta"]
     
    def get2d_idxs(self):
        #return [self.STATE_FST_LAD, self.STATE_FST_LAD + 1] 
        #return [self.STATE_FST_LAD + (self.conf.work.nlad - 1) * 2, self.STATE_FST_LAD + (self.conf.work.nlad - 1) * 2 + 1] 
        return self.PLOT_2D_INDXS #[self.STATE_X, self.STATE_Y] 
    

    def setup_traj_plot(self, ax: plt.Axes):
        ax.set(xlabel=r"$x$", ylabel=r"$y$")
        ax.scatter(
            self.cur_planner.waypoints[:, self.cur_planner.conf.wpt_xind], 
            self.cur_planner.waypoints[:, self.cur_planner.conf.wpt_yind], 
            color='blue', label='WPs', s = 0.01
            )
    
    def setup_traj2_plot(self, axes: list[plt.Axes]):
        # Plot the avoid set.
        axes[self.STATE_X].scatter(
            np.arange(len(self.cur_planner.waypoints)), self.cur_planner.waypoints[:, self.cur_planner.conf.wpt_xind],
            color='black', label='xs', s = 0.01
            )
        axes[self.STATE_Y].scatter(
            np.arange(len(self.cur_planner.waypoints)), self.cur_planner.waypoints[:, self.cur_planner.conf.wpt_yind],
            color='black', label='ys', s = 0.01
            )
        
            
    def grid_contour(self) -> tuple[BBFloat, BBFloat, TaskState]:
        # Contour with ( x axis=θ, y axis=H )
        n_xs = 2
        n_ys = 2
        b_xs = np.linspace(-0.1, 0.1, num=n_xs)
        b_ys = np.linspace(-0.1, 0.1, num=n_ys)

        x0 = jnp.zeros([self.nx])
        bb_x0 = ei.repeat(x0, "nx -> b1 b2 nx", b1=n_ys, b2=n_xs)

        bb_X, bb_Y = np.meshgrid(b_xs, b_ys)
        bb_x0 = bb_x0.at[:, :, self.STATE_X].set(bb_X)
        bb_x0 = bb_x0.at[:, :, self.STATE_Y].set(bb_Y)

        return bb_X, bb_Y, bb_x0
    
        state = self.reset()
        return state.reshape(1, *state.shape)

