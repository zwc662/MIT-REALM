import time
from f110_gym.envs.base_classes import Integrator
import yaml
import gym
import numpy as np
from argparse import Namespace

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from gmr import GMM

import joblib

import wandb
# Use wandb-core
wandb.require("core")
wandb.login()
wandb.init(
    # Set the project where this run will be logged
    project="basic-intro",
    # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
    name=f"experiment_rela_coord_test",
    # Track hyperparameters and run metadata
    )


from numba import njit

import pyglet
#from pyglet.gl import GL_POINTS

 

"""
Bicycle Model
"""



 
class BicycleModel:
    def __init__(self, wheelbase, dt = 0.01):
        self.wheelbase = wheelbase
        self.dt = dt

        # Initialize GP models
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1, (1e-2, 1e2))
        self.gp_x = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        self.gp_y = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

        # Initialize NN model 
        self.nn = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=500)


        # Initialize NN model 
        self.gmm = GMM(n_components=3, random_state = 0)
    
        # Data storage for GP
        self.clear_cache()
        
        self.X = None
        self.Y = None 

        self.load_models()
    
    def clear_cache(self):
        self.data = {'Xs': {'xs': [], 'ys': [], 'pose_theta': [], 'velocity_x': [], 'velocity_y': [], 'steering_angle': []},
                        'Ys': {'xs': [], 'ys': []}}
        
    def collect_data(self, pose_x, pose_y, pose_theta, velocity_x, velocity_y, steering_angle):
        beta = np.arctan(0.5 * np.tan(steering_angle))
        velocity = np.sqrt(velocity_x**2 + velocity_y**2)

        dx = velocity * np.cos(pose_theta + beta) * self.dt
        dy = velocity * np.sin(pose_theta + beta) * self.dt
        dtheta = (velocity / self.wheelbase) * np.sin(beta) * self.dt

        # Collect input
        self.data['Xs']['xs'].append(pose_x)
        self.data['Xs']['ys'].append(pose_y)
        self.data['Xs']['pose_theta'].append(pose_theta)
        self.data['Xs']['velocity_x'].append(velocity_x)
        self.data['Xs']['velocity_y'].append(velocity_y)
        self.data['Xs']['steering_angle'].append(steering_angle)
        
        # Correct Y in the previous step based on true state variables
        if len(self.data['Ys']['xs']) > 0:
            self.data['Ys']['xs'][-1] = pose_x - self.data['Ys']['xs'][-1]
        if len(self.data['Ys']['ys']) > 0:
            self.data['Ys']['ys'][-1] = pose_y - self.data['Ys']['ys'][-1]

        # Predict the next Y
        self.data['Ys']['xs'].append(pose_x + dx)
        self.data['Ys']['ys'].append(pose_y + dy)

        # Synch w/ wandb
        wandb.log({
                'x': self.data['Ys']['xs'][-1],
                'y': self.data['Ys']['ys'][-1],
                'pose_theta': self.data['Xs']['pose_theta'][-1],
                'velocity_x': self.data['Xs']['velocity_x'][-1],
                'velocity_y': self.data['Xs']['velocity_y'][-1],
                'bm_x': self.data['Ys']['xs'][-1],
                'bm_y': self.data['Ys']['ys'][-1],
                'res_x': self.data['Ys']['xs'][-2] if len(self.data['Ys']['xs']) > 1 else 0,
                'res_y': self.data['Ys']['ys'][-2] if len(self.data['Ys']['ys']) > 1 else 0
            }
        )
    


 
    def update_models(self, test_size = 0.5, num_training = 500, random_seed = 100):
        X = np.array([#self.data['Xs']['xs'],
                      #self.data['Xs']['ys'],
                      self.data['Xs']['pose_theta'],
                      self.data['Xs']['velocity_x'],
                      self.data['Xs']['velocity_y'],
                      self.data['Xs']['steering_angle']]).T
        Y = np.array([self.data['Ys']['xs'],
                      self.data['Ys']['ys']]).T
        
        if self.X is None and self.Y is None:
            self.X = X[1:]
            self.Y = Y[1:]
        else:
            self.X = np.vstack((self.X, X))
            self.Y = np.vstack((self.Y, Y))
        self.clear_cache()

        
        assert X.shape[0] > 20
        
        X_train, X_test, y_train, y_test = train_test_split(self.X[1:-10], self.Y[1:-10], test_size=test_size, random_state=random_seed)

        #self.train_gp(X_train[: num_training], y_train[-num_training:])
        #self.train_nn(X_train[: num_training], y_train[-num_training:])
        #self.train_gmm(X_train[: num_training], y_train[-num_training:])

        self.test_gp(X_test, y_test)
        self.test_nn(X_test, y_test)
        self.test_gmm(X_test, y_test)

        self.save_models()

    def train_gmm(self, X_train, Y_train, n_components=2):
        # Combine X_train and Y_train for GMM fitting
        data = np.hstack((X_train, Y_train))
        self.gmm.from_samples(data)
    
    def test_gmm(self, X_test, Y_test):
        # Predict the conditional mean of Y given X for GMM
        preds = self.gmm.predict(np.arange(X_test.shape[-1]), X_test)
        mse_x = np.mean((preds[:, 0] - Y_test[:, 0]) ** 2)
        mse_y = np.mean((preds[:, 1] - Y_test[:, 1]) ** 2)
        
        print(f'GMM Model Mean Squared Error (x): {mse_x}')
        print(f'GMM Model Mean Squared Error (y): {mse_y}')

        wandb.log({
            'gmm_x_mse': mse_x,
            'gmm_y_mse': mse_y
        })
        
        
        return mse_x, mse_y

         
    def train_gp(self, X_train, y_train):
        y_train_x = y_train[:, 0]
        y_train_y = y_train[:, 1]
         
        self.gp_x.fit(X_train, y_train_x)
        self.gp_y.fit(X_train, y_train_y)
        
        # Clear data after updating
        #self.data = {'Xs': {'xs': [], 'ys': [], 'pose_theta': [], 'velocity': [], 'steering_angle': []},
        #                'Ys': {'x': [], 'y': [], 'theta': []}}
    

    def test_gp(self, X_test, Y_test):
        preds_x = self.gp_x.predict(X_test)
        preds_y = self.gp_y.predict(X_test)
        
        mse_x = np.mean((preds_x - Y_test[:, 0]) ** 2)
        mse_y = np.mean((preds_y - Y_test[:, 1]) ** 2)
        
        print(f'GP Model Mean Squared Error (x): {mse_x}')
        print(f'GP Model Mean Squared Error (y): {mse_y}')

        wandb.log({
            'gp_x_mse': mse_x,
            'gp_y_mse': mse_y
        })
        
        return mse_x, mse_y
        


    def train_nn(self, X_train, y_train):
        self.nn.fit(X_train, y_train)
    
    def test_nn(self, X_test, y_test):
        preds = self.nn.predict(X_test)
        
        mse_x = np.mean((preds[:, 0] - y_test[:, 0]) ** 2)
        mse_y = np.mean((preds[:, 1] - y_test[:, 1]) ** 2)
        
        
        print(f'NN Model Mean Squared Error (x): {mse_x}')
        print(f'NN Model Mean Squared Error (y): {mse_y}')

        wandb.log({
            'nn_x_mse': mse_x,
            'nn_y_mse': mse_y
        })
        
        return mse_x, mse_y
  
    def save_models(self, gp_x_file='gp_x.pkl', gp_y_file='gp_y.pkl', nn_file='nn.pkl', gmm_file='gmm.pkl'):
        joblib.dump(self.gp_x, gp_x_file)
        joblib.dump(self.gp_y, gp_y_file)
        joblib.dump(self.nn, nn_file)
        joblib.dump(self.gmm, gmm_file)
        print(f"Models saved to {gp_x_file}, {gp_y_file}, {nn_file}, {gmm_file}.")

    def load_models(self, gp_x_file='gp_x.pkl', gp_y_file='gp_y.pkl', nn_file='nn.pkl', gmm_file='gmm.pkl'):
        self.gp_x = joblib.load(gp_x_file)
        self.gp_y = joblib.load(gp_y_file)
        self.nn = joblib.load(nn_file)
        self.gmm = joblib.load(gmm_file)
        print(f"Models loaded from {gp_x_file}, {gp_y_file}, and {nn_file}.")




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
    l2s   = diffs[:,0]**2 + diffs[:,1]**2
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

class PurePursuitPlanner:
    """
    Example Planner
    """
    def __init__(self, conf, wb):
        self.wheelbase = wb
        self.conf = conf
        self.load_waypoints(conf)
        self.max_reacquire = 20.

        self.drawn_waypoints = []
        

        self.bicycle_model = BicycleModel(wb)
        

    def load_waypoints(self, conf):
        """
        loads waypoints
        """
        self.waypoints = np.loadtxt(conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip)

    def render_waypoints(self, e):
        """
        update waypoints being drawn by EnvRenderer
        """

        #points = self.waypoints

        points = np.vstack((self.waypoints[:, self.conf.wpt_xind], self.waypoints[:, self.conf.wpt_yind])).T
        
        scaled_points = 50.*points

        for i in range(points.shape[0]):
            if len(self.drawn_waypoints) < points.shape[0]:
                b = e.batch.add(1, GL_POINTS, None, ('v3f/stream', [scaled_points[i, 0], scaled_points[i, 1], 0.]),
                                ('c3B/stream', [183, 193, 222]))
                self.drawn_waypoints.append(b)
            else:
                self.drawn_waypoints[i].vertices = [scaled_points[i, 0], scaled_points[i, 1], 0.]
     
                 
    def _get_current_waypoint(self, waypoints, lookahead_distance, position, theta):
        """
        gets the current waypoint to follow
        """
        wpts = np.vstack((self.waypoints[:, self.conf.wpt_xind], self.waypoints[:, self.conf.wpt_yind])).T
        nearest_point, nearest_dist, t, i = nearest_point_on_trajectory(position, wpts)
        speed = 5 * (1 - np.exp(-t))
        if nearest_dist < lookahead_distance:
            lookahead_point, i2, t2 = first_point_on_trajectory_intersecting_circle(position, lookahead_distance, wpts, i+t, wrap=True)
            if i2 == None:
                return None
            current_waypoint = np.empty((3, ))
            # x, y
            current_waypoint[0:2] = wpts[i2, :]
            # speed
            current_waypoint[2] = speed #waypoints[i, self.conf.wpt_vind]
            return current_waypoint
        elif nearest_dist < self.max_reacquire:
            return np.append(wpts[i, :], speed) #waypoints[i, self.conf.wpt_vind])
        else:
            return None

    def plan(self, obs, work):
        pose_x = obs['poses_x'][0]
        pose_y = obs['poses_y'][0]
        pose_theta = obs['poses_theta'][0] 
        velocity_x = obs['linear_vels_x'][0]
        velocity_y = obs['linear_vels_y'][0]

        lookahead_distance = work['tlad']
        vgain = work['vgain']
        
        """
        gives actuation given observation
        """
        position = np.array([pose_x, pose_y])
        lookahead_point = self._get_current_waypoint(self.waypoints, lookahead_distance, position, pose_theta)

        if lookahead_point is None:
            return 4.0, 0.0

        speed, steering_angle = get_actuation(pose_theta, lookahead_point, position, lookahead_distance, self.wheelbase)
        speed = vgain * speed
        
        self.bicycle_model.collect_data(pose_x, pose_y, pose_theta, velocity_x, velocity_y, steering_angle)
    
        return speed, steering_angle
    

    def call_back(self):
        print("Call back training and testing")
        # Training and Testing
        self.bicycle_model.update_models()
        

class FlippyPlanner:
    """
    Planner designed to exploit integration methods and dynamics.
    For testing only. To observe this error, use single track dynamics for all velocities >0.1
    """
    def __init__(self, speed=1, flip_every=1, steer=2):
        self.speed = speed
        self.flip_every = flip_every
        self.counter = 0
        self.steer = steer
    
    def render_waypoints(self, *args, **kwargs):
        pass

    def plan(self, *args, **kwargs):
        if self.counter%self.flip_every == 0:
            self.counter = 0
            self.steer *= -1
        return self.speed, self.steer


def main():
    """
    main entry point
    """
    
    work = {'mass': 3.463388126201571, 'lf': 0.15597534362552312, 'tlad': 0.82461887897713965, 'vgain': 1.375}#0.90338203837889}
    
    with open('config_example_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    planner = PurePursuitPlanner(conf, (0.17145+0.15875)) #FlippyPlanner(speed=0.2, flip_every=1, steer=10)

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

        planner.render_waypoints(env_renderer) 

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
        obs, step_reward, done, info = env.step(np.array([[steer, speed]]))
        laptime += step_reward
        #env.render(mode='human')
        if step > 100 and step % 100 == 1:
            print(f'Step {step}')
            planner.call_back()

        step += 1
        
    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start)

if __name__ == '__main__':
    main()
