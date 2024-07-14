import math
import gym
from gym.utils import seeding
from lib.model_learner import OnlineModelLearner
from lib.mat_engine import MatEngine
from numpy import linalg as LA
# from matplotlib.collections import PathCollection
# from matplotlib.legend_handler import HandlerPathCollection, HandlerLine2D
from test.phase_plotter import PhasePlotter
import matplotlib.pyplot as plt
from numpy.linalg import inv
import numpy as np
import copy
import os

## work 0.6/0.55/0.5/0.45/0.4(1.009)

# MATRIX_A = np.array([[1, 0.03333333, 0, 0],
#                      [0.0247, 1.1204, 1.1249, 0.2339],
#                      [0, 0, 1, 0.03333333],
#                      [-0.0580, -0.2822, -1.8709, 0.4519]])

MATRIX_A = np.array([[1.0000, 0.0333, 0, 0],
                     [0.6465, 1.5268, 2.1666, 0.4020],
                     [0, 0, 1.0000, 0.0333],
                     [-1.5151, -1.2348, -4.3123, 0.0577]])

MATRIX_P = np.array([[11.7610, 6.1422, 14.7641, 3.1142],
                     [6.1422, 3.8248, 9.3641, 1.9903],
                     [14.7641, 9.3641, 25.9514, 5.0703],
                     [3.1142, 1.9903, 5.0703, 1.0949]]) * 0.6

MATRIX_S = np.array([[1.0000, 0.0333, 0, 0],
                     [0.8710, 1.6804, 2.6660, 0.4787],
                     [0, 0, 1.0000, 0.0333],
                     [-2.0415, -1.5946, -5.4828, -0.1219]])

F = np.array([26.0660, 20.3602, 81.4709, 14.3251]) * 0.40


# MATRIX_P = np.array([[13.6303, 7.0668, 16.7955, 3.5460],
#                      [7.0668, 4.3791, 10.5976, 2.2520],
#                      [16.7955, 10.5976, 29.1577, 5.6641],
#                      [3.5460, 2.2520, 5.6641, 1.2207]])
#
# MATRIX_S = np.array([[1.0000, 0.0333, 0, 0],
#                      [0.9547, 1.7423, 2.8584, 0.5107],
#                      [0, 0, 1.0000, 0.0333],
#                      [-2.2375, -1.7398, -5.9336, -0.1971]])
#
# F = np.array([28.5681, 22.2139, 87.2267, 15.2840]) * 0.45


# stable but unsafe
# MATRIX_A = np.array([[1, 0.03333333, 0, 0],
#                      [0.0247, 1.1204, 1.1249, 0.2339],
#                      [0, 0, 1, 0.03333333],
#                      [-0.0580, -0.2822, -1.8709, 0.4519]])
#
# MATRIX_P = np.array([[20.9142, 9.2155, 21.1088, 4.5534],
#                      [9.2155, 4.6313, 10.7144, 2.3333],
#                      [21.1088, 10.7144, 26.3303, 5.5216],
#                      [4.5534, 2.3333, 5.5216, 1.2113]])
#
# MATRIX_S = np.array([[1.0000, 0.0333, 0, 0],
#                      [1.5990, 2.0467, 3.4598, 0.6621],
#                      [0, 0, 1.0000, 0.0333],
#                      [-3.7476, -2.4531, -7.3432, -0.5518]])
#
# F = np.array([47.8498, 31.3218, 105.2240, 19.8128]) * 0.35


class CartpoleParams:
    def __init__(self):
        self.phase_plot = False
        self.phase_dir = "plots/phases/demo"
        self.trajectory_plot = False
        self.trajectory_dir = "plots/trajectories/demo"
        self.simplex_enable = True
        self.simplex_learn = True
        self.continue_learn = False
        self.dwell_steps = 100
        self.chi = 0.5
        self.eval_epsilon = 0.3
        self.simplex_epsilon = 0.6
        self.x_threshold = 0.9
        self.x_dot_threshold = 15
        self.theta_threshold = 0.8
        self.theta_dot_threshold = 15
        self.force_threshold = 30
        self.kinematics_integrator = 'euler'
        self.gravity = 9.8
        self.mass_cart = 0.94
        self.mass_pole = 0.23
        self.force_mag = 10.0
        self.voltage_mag = 5.0
        self.length = 0.64
        self.theta_random_std = 0.8
        self.friction_cart = 10
        self.friction_pole = 0.0011
        self.simulation_frequency = 30
        self.with_friction = True
        self.force_input = True
        self.random_noise = {
            "actuator": {
                "apply": True,
                "type": "gaussian",
                "mean": 0,
                "stddev": 2
            },
            "friction": {
                "cart": {
                    "apply": True,
                    "type": "gaussian",
                    "mean": 0,
                    "stddev": 2
                },
                "pole": {
                    "apply": True,
                    "type": "gaussian",
                    "mean": 0,
                    "stddev": 2
                }
            }
        }
        self.ini_states = [0.1, 0.1, 0.15, 0.0, False]
        self.targets = [0., 0.]
        self.distance_score_factor = 0
        self.tracking_error_factor = 1
        self.lyapunov_reward_factor = 1
        self.high_performance_reward_factor = 0.5
        self.action_penalty = 0
        self.crash_penalty = 0
        self.observe_reference_states = False
        self.random_reset_train = True
        self.random_reset_eval = False
        self.update_reference_model = True
        self.sparse_reset = False
        self.use_ubc_lya_reward = True
        self.use_termination = True
        self.n_points_per_dim = 10


class Cartpole(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, params: CartpoleParams):

        self.params = params
        self.total_mass = (self.params.mass_cart + self.params.mass_pole)
        self.half_length = self.params.length * 0.5
        self.pole_mass_length_half = self.params.mass_pole * self.half_length
        self.tau = 1 / self.params.simulation_frequency

        self.seed()
        self.viewer = None
        self.states = None
        self.steps_beyond_terminal = None

        self.states_dim = 4  # x, x_dot, theta, theta_dot
        self.states_observations_dim = 5  # x, x_dot, s_theta, c_theta, theta_dot
        self.states_observations_refer_dim = 4  # error between the states of ref and real
        self.action_dim = 1  # force input or voltage
        self.states_refer = None
        self.reward_list = []

        # Phase Plotter
        self.p_mat = MATRIX_P
        self.plotter = PhasePlotter(
            p_mat=self.p_mat,
            phase_dir=self.params.phase_dir,
            trajectory_dir=self.params.trajectory_dir,
            simplex_enable=self.params.simplex_enable
        )

        self.ut = None
        self.state_list = []
        self.force_list = []
        self.safety_val_list = []

        # MATLAB Engine
        self.mat_engine = MatEngine()
        self.F_hat = np.zeros((1, 4)).reshape(1, 4)

        # Online Model Learner
        self.model_learner = OnlineModelLearner(window_size=8, sample_freq=self.params.simulation_frequency)
        self.pP, self.vP = self.get_pP_and_vP()

        # For Simplex
        self.last_action_mode = None
        self.eq_point = None
        self.Ak = None
        self.Bk = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def refer_step(self):
        x, x_dot, theta, theta_dot, _ = self.states_refer
        current_states = np.transpose([x, x_dot, theta, theta_dot])  # 4 x 1
        next_states = MATRIX_A @ current_states
        x, x_dot, theta, theta_dot = np.squeeze(next_states).tolist()
        theta_rescale = math.atan2(math.sin(theta), math.cos(theta))  # to rescale theta into [-pi, pi)
        # failed = self.is_failed(x, theta_dot)
        failed = self.is_failed(x, theta)
        new_states = [x, x_dot, theta_rescale, theta_dot, failed]
        self.states_refer = new_states  # to update animation
        return self.states_refer

    def step(self, action: float, action_mode="residual"):
        """
        param: action: a scalar value (not numpy type) [-1,1]
        return: a list of states
        """
        # action = 0
        nominal_drl_action = action
        x, x_dot, theta, theta_dot, _ = self.states
        force = action * self.params.force_mag  # 2.5  # -10, 10
        if action_mode == 'residual':
            force_res = F[0] * x + F[1] * x_dot + F[2] * theta + F[3] * theta_dot  # residual control commands
            force = force + force_res  # RL control commands + residual control commands
        elif action_mode == 'drl':
            force *= 3
        elif action_mode == 'model':
            f_random = np.random.uniform(-35, 35)
            force = F[0] * x + F[1] * x_dot + F[2] * theta + F[3] * theta_dot + f_random

            print(f"model-based force: {force}")
            print(f"random force: {f_random}")
        elif action_mode == 'simplex':

            # s_eq = np.array([0.14998542, 4.84414696, -0.25146283, -5.87739881])
            # s_eq = np.array([0.23434349, 0, -0.22644896, 0])
            # s_eq = np.array([0, 0, 0.22644896, 0])
            # s_eq = np.array([0.23434349, 4.50993478, -0.22644896, -4.78591104])

            s = np.array([x, x_dot, theta, theta_dot])
            sbar_star = self.eq_point

            e = s - sbar_star

            redundancy_term = sbar_star - self.Ak @ sbar_star

            print(f"redundancy term: {redundancy_term}")
            print(redundancy_term.shape)
            print(f"Bk: {self.Bk}")

            v1 = np.squeeze(redundancy_term[1] / self.Bk[1])
            v2 = np.squeeze(redundancy_term[3] / self.Bk[3])
            # v = np.linalg.pinv(self.Bk).squeeze() @ (np.eye(4) - self.Ak) @ sbar_star
            v = (v1 + v2) / 2
            # print(f"v1: {v1}")
            # print(f"v2: {v2}")
            print(f"v is: {v}")
            print(f"s is: {s}")

            # print(f"F_hat: {self.F_hat}")
            force = self.F_hat @ e + v
            # force = K @ e + v
            print(f"simplex force: {force}")

            # Use simplex to learn
            if self.params.simplex_learn:
                nominal_drl_action = (force - (
                        F[0] * x + F[1] * x_dot + F[2] * theta + F[3] * theta_dot)) / self.params.force_mag
            # Only for safety guarantee
            else:
                pass

        else:
            raise RuntimeError(f'Uncleared action mode for agent: {action_mode}')

        self.last_action_mode = action_mode

        # print(f"current x: {x}")
        # print(f"current theta_dot: {theta_dot}")
        f_min = -self.params.force_threshold
        f_max = self.params.force_threshold
        force = np.clip(force, a_min=f_min, a_max=f_max)

        # Actual force applied to plant after random noise
        if self.params.random_noise.actuator.apply:
            mean = self.params.random_noise.actuator.mean
            stddev = self.params.random_noise.actuator.stddev
            force += np.random.normal(loc=mean, scale=stddev)

        print(f"applied force is: {force}")

        self.ut = force
        self.force_list.append(self.ut)

        # Data Record into file
        # xx = np.asarray(self.states[:4])
        # uu = np.asarray(self.ut)
        # f = open("matrix/trajectories.txt", "a+")
        # # ctn = f"xi: {xx}, ui: {uu}\n"
        # data = np.hstack((xx, uu)).reshape(1, 5)
        # print(f"data: {data}")
        # np.savetxt(f, data, delimiter=',', fmt='%.5f')
        # # f.write(ctn)
        # f.close()

        # Update Model Learner
        # self.model_learner.sample(xi=xx, ui=uu)

        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # kinematics of the inverted pendulum
        if self.params.with_friction:
            """ with friction"""
            temp \
                = (force + self.pole_mass_length_half * theta_dot ** 2 *
                   sintheta - self.params.friction_cart * x_dot) / self.total_mass

            thetaacc = \
                (self.params.gravity * sintheta - costheta * temp -
                 self.params.friction_pole * theta_dot / self.pole_mass_length_half) / \
                (self.half_length * (4.0 / 3.0 - self.params.mass_pole * costheta ** 2 / self.total_mass))
            xacc = temp - self.pole_mass_length_half * thetaacc * costheta / self.total_mass

        else:
            """without friction"""
            temp = (force + self.pole_mass_length_half * theta_dot ** 2 * sintheta) / self.total_mass
            thetaacc = (self.params.gravity * sintheta - costheta * temp) / \
                       (self.half_length * (4.0 / 3.0 - self.params.mass_pole * costheta ** 2 / self.total_mass))
            xacc = temp - self.pole_mass_length_half * thetaacc * costheta / self.total_mass

        if self.params.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc  # here we inject disturbances
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc  # here we inject disturbances
            # failed = self.is_failed(x, theta_dot)
            failed = self.is_failed(x, theta)

        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot
            # failed = self.is_failed(x, theta_dot)
            failed = self.is_failed(x, theta)

        # print(f"update x: {x}")
        # print(f"updated theta_dot: {theta_dot}")

        theta_rescale = math.atan2(math.sin(theta), math.cos(theta))  # wrap to [-pi, pi]
        new_states = [x, x_dot, theta_rescale, theta_dot, failed]
        print(f"curr state: {self.states}")
        print(f"new state: {new_states}")
        self.states = new_states  # to update animation
        return self.states, nominal_drl_action

    def reset(self, reset_states=None):
        # print(f"<====== Env Reset: Reset at predefined condition =====>")
        if reset_states is not None:
            self.states = reset_states
            self.states_refer = reset_states
        else:
            self.states = self.params.ini_states
            self.states_refer = self.states

    def random_reset(self, mode='train'):
        print("<====== Env Reset: Random =====>")
        # threshold = self.params.epsilon2hpc

        flag = True
        while flag:
            ran_x = np.random.uniform(-0.9, 0.9)
            ran_v = np.random.uniform(-3.0, 3.0)
            # ran_v = np.random.uniform(-2.0, 2.0)
            ran_theta = np.random.uniform(-0.8, 0.8)
            ran_theta_v = np.random.uniform(-4.5, 4.5)
            # ran_theta_v = np.random.uniform(-2.5, 2.5)
            # ran_theta_v = np.random.uniform(-3, 3)

            # state_vec = np.array([ran_x, ran_theta])

            safety_val = self.safety_value(
                states=np.array([ran_x, ran_v, ran_theta, ran_theta_v]), p_mat=MATRIX_P
            )

            # safety_val = self.safety_value(states=state_vec, p_mat=self.pP)
            if safety_val < self.params.simplex_epsilon:
                flag = False

            # if mode is not 'train' and safety_val <= self.params.eval_epsilon:
            #     flag = True

        failed = False
        self.states = [ran_x, ran_v, ran_theta, ran_theta_v, failed]
        # self.states = [ran_x, 0, ran_theta, 0, failed]
        # self.states = [0.50680287, 0.83706794, -0.02444331, 1.23702147, failed]
        # self.states = [-0.07104200950754436, 0.3039070583174803, -0.10771449874003003, 3.369696410916341, failed]
        # self.states = [-0.504200950754436, 0.4039070583174803, 0.24771449874003003, 3.369696410916341, failed]
        # self.states = [-0.46386487469408112, 0, 0.2278973388408276276, 0, failed]
        # self.states = [0, 0, -0.1578973388408276276, 0, failed]
        # state_vec = np.array([-0.46386487469408112, 0.2278973388408276276])
        # v = self.safety_value(states=state_vec, p_mat=self.pP)

        # print(f"state safety val:{v}")
        self.states_refer = copy.deepcopy(self.states)

    def render(self, mode='human', states=None, is_normal_operation=True):

        screen_width = 600
        screen_height = 400
        world_width = self.params.x_threshold * 2 + 1
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * self.params.length
        cartwidth = 50.0
        cartheight = 30.0
        target_width = 45
        target_height = 45

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.targettrans = rendering.Transform()
            target = rendering.Image('./lib/env/target.svg', width=target_width, height=target_height)
            target.add_attr(self.targettrans)
            self.viewer.add_geom(target)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            if not is_normal_operation:
                cart.set_color(1.0, 0, 0)
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)
            self._pole_geom = pole

        if states is None:
            if self.states is None:
                return None
            else:
                x = self.states
        else:
            x = states

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        targetx = 0 * scale + screen_width / 2.0
        targety = polelen + carty

        self.carttrans.set_translation(cartx, carty)
        self.targettrans.set_translation(targetx, targety)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    # def is_failed(self, x, theta_dot):
    #     failed = bool(x <= -self.params.x_threshold
    #                   or x >= self.params.x_threshold
    #                   or theta_dot > self.params.theta_dot_threshold)
    #     return failed

    def is_failed(self, x, theta):
        failed = bool(x <= -self.params.x_threshold
                      or x >= self.params.x_threshold
                      or theta >= self.params.theta_threshold
                      or theta <= -self.params.theta_threshold)
        return failed

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def get_distance_score(self, observations, target):
        distance_score_factor = 5  # to adjust the exponential gradients
        cart_position = observations[0]
        pendulum_angle_sin = observations[2]
        pendulum_angle_cos = observations[3]

        target_cart_position = target[0]
        target_pendulum_angle = target[1]

        pendulum_length = self.params.length

        pendulum_tip_position = np.array(
            [cart_position + pendulum_length * pendulum_angle_sin, pendulum_length * pendulum_angle_cos])

        target_tip_position = np.array(
            [target_cart_position + pendulum_length * np.sin(target_pendulum_angle),
             pendulum_length * np.cos(target_pendulum_angle)])

        distance = np.linalg.norm(target_tip_position - pendulum_tip_position)

        distance_score = np.exp(-distance * distance_score_factor)
        return distance_score

    @staticmethod
    def get_lyapunov_reward(p_matrix, states_real):
        state = np.array(states_real[0:4])
        state = np.expand_dims(state, axis=0)
        Lya1 = np.matmul(state, p_matrix)
        Lya = np.matmul(Lya1, np.transpose(state))
        return Lya

    @staticmethod
    def get_tracking_error(p_matrix, states_real, states_reference):

        state = np.array(states_real[0:4])
        state = np.expand_dims(state, axis=0)
        state_ref = np.array(states_reference[0:4])
        state_ref = np.expand_dims(state_ref, axis=0)

        state_error = state - state_ref
        eLya1 = np.matmul(state_error, p_matrix)
        eLya = np.matmul(eLya1, np.transpose(state_error))

        error = -eLya

        return error

    def test_safety(self, states_current, action, states_next):
        miu_1 = 4.0
        miu_2 = 0.2
        alfa = 0.85
        beta_min = ((miu_1 - 1) * (1 - miu_2) / alfa) - miu_2

        observations, _ = states2observations(states_current)

        ##########
        tem_state_a = np.array(states_current[0:4])
        tem_state_b = np.expand_dims(tem_state_a, axis=0)
        tem_state_c = np.matmul(tem_state_b, np.transpose(MATRIX_S))
        tem_state_d = np.matmul(tem_state_c, MATRIX_P)
        lyapunov_reward_current_aux = np.matmul(tem_state_d, np.transpose(tem_state_c))
        #########

        lyapunov_reward_next = self.get_lyapunov_reward(MATRIX_P, states_next)
        # if lyapunov_reward_next > 1:
        #     print("lyapunov_reward_next", lyapunov_reward_next)
        gamma = 1.0
        upper_lyapunov_reward_next = miu_1 * lyapunov_reward_current_aux
        beta_lyapunov_reward_next = upper_lyapunov_reward_next - lyapunov_reward_next
        cond21 = (1 - miu_1) * lyapunov_reward_current_aux + beta_lyapunov_reward_next * gamma
        cond17 = upper_lyapunov_reward_next - lyapunov_reward_next
        control = cond17 * gamma - beta_min
        theta = (lyapunov_reward_current_aux / 1) + (control / (1 * (1 - miu_1)))

        # print(f"---theta---  value: {theta} result:{theta >= alfa}")
        # print(f"---cond21--- value: {cond21} result:{cond21 >= (alfa - 1)}")
        print(f"---reward_next--- value: {lyapunov_reward_next} result:{lyapunov_reward_next <= 1}")
        # print("")

        test_result = {"theta_test": theta >= alfa,
                       "cond21": cond21 >= (alfa - 1),
                       "reward_next": lyapunov_reward_next <= 1}

        return test_result

    def reward_fcn(self, states_current, action, states_next):

        observations, _ = states2observations(states_current)
        targets = self.params.targets  # [0, 0] stands for position and angle

        distance_score = self.get_distance_score(observations, targets)
        distance_reward = distance_score * self.params.high_performance_reward_factor

        lyapunov_reward_current = self.get_lyapunov_reward(MATRIX_P, states_current)

        ##########
        tem_state_a = np.array(states_current[0:4])
        tem_state_b = np.expand_dims(tem_state_a, axis=0)
        tem_state_c = np.matmul(tem_state_b, np.transpose(MATRIX_S))
        tem_state_d = np.matmul(tem_state_c, MATRIX_P)
        lyapunov_reward_current_aux = np.matmul(tem_state_d, np.transpose(tem_state_c))
        ##########

        lyapunov_reward_next = self.get_lyapunov_reward(MATRIX_P, states_next)

        if self.params.use_ubc_lya_reward:
            lyapunov_reward = lyapunov_reward_current - lyapunov_reward_next
        else:
            lyapunov_reward = lyapunov_reward_current_aux - lyapunov_reward_next  # phy-drl

        self.reward_list.append(np.squeeze(lyapunov_reward))

        lyapunov_reward *= self.params.lyapunov_reward_factor
        action_penalty = -1 * self.params.action_penalty * action * action
        r = distance_reward + lyapunov_reward + action_penalty

        return r, distance_score

    def get_pP_and_vP(self):
        P = MATRIX_P
        pP = np.zeros((2, 2))
        vP = np.zeros((2, 2))

        # For velocity
        vP[0][0] = P[1][1]
        vP[1][1] = P[3][3]
        vP[0][1] = P[1][3]
        vP[1][0] = P[1][3]

        # For position
        pP[0][0] = P[0][0]
        pP[1][1] = P[2][2]
        pP[0][1] = P[0][2]
        pP[1][0] = P[0][2]

        return pP, vP

    def safety_value(self, states, p_mat):
        return states.transpose() @ p_mat @ states

    def get_A_B_mat_by_state(self, state: np.ndarray):
        x = state[0]
        x_dot = state[1]
        theta = state[2]
        theta_dot = state[3]
        # print(f"final_theta: {theta}")
        # print(f"theta_dot: {theta_dot}")

        A = np.zeros((4, 4))
        A[0][1] = 1
        A[2][3] = 1

        mc = self.params.mass_cart
        mp = self.params.mass_pole
        g = self.params.gravity
        l = self.params.length / 2

        term = 4 / 3 * (mc + mp) - mp * np.cos(theta) * np.cos(theta)

        A[1][2] = -mp * g * np.sin(theta) * np.cos(theta) / (theta * term)
        A[1][3] = 4 / 3 * mp * l * np.sin(theta) * theta_dot / term
        A[3][2] = g * np.sin(theta) * (mc + mp) / (l * theta * term)
        A[3][3] = -mp * np.sin(theta) * np.cos(theta) * theta_dot / term

        B = np.zeros((4, 1))
        B[1] = 4 / 3 / term
        B[3] = -np.cos(theta) / (l * term)
        return A, B

    def get_A_B_mat_by_taylor(self, theta, theta_dot, u):
        A = np.zeros((4, 4))
        A[0][1] = 1
        A[2][3] = 1

        mc = self.params.mass_cart
        mp = self.params.mass_pole
        g = self.params.gravity
        l = self.params.length / 2

        A[1][2] = ((np.cos(theta) * (
                g * np.cos(theta) + (np.sin(theta) * (l * mp * np.sin(theta) * theta_dot ** 2 + u)) / (mc + mp) - (
                l * mp * theta_dot ** 2 * np.cos(theta) ** 2) / (mc + mp))) / (
                           l * ((mp * np.cos(theta) ** 2) / (mc + mp) - 4 / 3)) + l * mp * theta_dot ** 2 * np.cos(
            theta) - (
                           np.sin(theta) * (
                           g * np.sin(theta) - (np.cos(theta) * (l * mp * np.sin(theta) * theta_dot ** 2 + u)) / (
                           mc + mp))) / (
                           l * ((mp * np.cos(theta) ** 2) / (mc + mp) - 4 / 3)) + (
                           2 * mp * np.cos(theta) ** 2 * np.sin(theta) * (
                           g * np.sin(theta) - (np.cos(theta) * (l * mp * np.sin(theta) * theta_dot ** 2 + u)) / (
                           mc + mp))) / (
                           l * (mc + mp) * ((mp * np.cos(theta) ** 2) / (mc + mp) - 4 / 3) ** 2)) / (mc + mp)

        A[1][3] = (2 * l * mp * theta_dot * np.sin(theta) - (
                2 * mp * theta_dot * np.cos(theta) ** 2 * np.sin(theta)) / (
                           (mc + mp) * ((mp * np.cos(theta) ** 2) / (mc + mp) - 4 / 3))) / (mc + mp)

        A[3][2] = -(g * np.cos(theta) + (np.sin(theta) * (l * mp * np.sin(theta) * theta_dot ** 2 + u)) / (mc + mp) - (
                l * mp * theta_dot ** 2 * np.cos(theta) ** 2) / (mc + mp)) / (
                          l * ((mp * np.cos(theta) ** 2) / (mc + mp) - 4 / 3)) - (
                          2 * mp * np.cos(theta) * np.sin(theta) * (
                          g * np.sin(theta) - (np.cos(theta) * (l * mp * np.sin(theta) * theta_dot ** 2 + u)) / (
                          mc + mp))) / (
                          l * (mc + mp) * ((mp * np.cos(theta) ** 2) / (mc + mp) - 4 / 3) ** 2)

        A[3][3] = (2 * mp * theta_dot * np.cos(theta) * np.sin(theta)) / (
                (mc + mp) * ((mp * np.cos(theta) ** 2) / (mc + mp) - 4 / 3))

        B = np.zeros((4, 1))
        B[1] = -(np.cos(theta) ** 2 / (l * (mc + mp) * ((mp * np.cos(theta) ** 2) / (mc + mp) - 4 / 3)) - 1) / (mc + mp)
        B[3] = np.cos(theta) / (l * (mc + mp) * ((mp * np.cos(theta) ** 2) / (mc + mp) - 4 / 3))

        return A, B

    def get_discrete_A_and_B(self, Ac, Bc):
        T = 1 / self.params.simulation_frequency
        Ak = Ac * T + np.eye(4)
        Bk = Bc * T
        return Ak, Bk

    def calculate_F(self, Ac: np.ndarray, Bc: np.ndarray):
        # Current state
        Sc = np.array([[self.states[0], self.states[1], self.states[2], self.states[3]]])

        # Desired state
        Sd = self.eq_point
        Sd = self.params.chi * Sc
        # Sd = np.array([[0, 0, 0, 0]])

        # Discrete A and B
        Ak, Bk = self.get_discrete_A_and_B(Ac=Ac, Bc=Bc)

        # Call Matlab Engine for F_hat
        F_hat = self.mat_engine.feedback_law(Ac, Bc, Ak, Bk, Sc, Sd)
        # K = self.mat_engine.feedback_control(Ac, Bc, Ak, Bk, Sc, Sd)

        self.F_hat = np.asarray(F_hat).squeeze()


def states2observations(states):
    x, x_dot, theta, theta_dot, failed = states
    observations = [x, x_dot, math.sin(theta), math.cos(theta), theta_dot]
    return observations, failed


def observations2states(observations, failed):
    x, x_dot, s_theta, c_theta, theta_dot = observations[:5]
    states = [x, x_dot, np.arctan2(s_theta, c_theta), theta_dot, failed]
    return states


def get_init_condition(n_points_per_dim=20):
    eigen_values, eigen_vectors = np.linalg.eig(MATRIX_P)  # get eigen value and eigen vector

    Q = eigen_vectors

    initial_condition_list = []

    for i in range(n_points_per_dim):
        angle_1 = i * math.pi / n_points_per_dim
        y0 = math.sqrt(1 / eigen_values[0]) * math.cos(angle_1)
        vector_in_3d = math.sin(angle_1)

        if vector_in_3d == 0:
            y1 = 0
            y2 = 0
            y3 = 0
            s = Q @ np.array([y0, y1, y2, y3]).transpose()
            # print(s.transpose() @ P_matrix_4 @ s)
            initial_condition_list.append([s[0], s[1], s[2], s[3], False])
            continue

        for k in range(n_points_per_dim):
            angle_2 = k * math.pi / n_points_per_dim
            y1 = vector_in_3d * math.sqrt(1 / eigen_values[1]) * math.cos(angle_2)
            vector_in_2d = vector_in_3d * math.sin(angle_2)

            if vector_in_2d == 0:
                y2 = 0
                y3 = 0
                s = Q @ np.array([y0, y1, y2, y3]).transpose()
                # print(s.transpose() @ P_matrix_4 @ s)
                initial_condition_list.append([s[0], s[1], s[2], s[3], False])
                continue

            for j in range(n_points_per_dim):
                angle_3 = j * math.pi / n_points_per_dim
                y2 = vector_in_2d * math.sqrt(1 / eigen_values[2]) * math.cos(angle_3)
                y3 = vector_in_2d * math.sqrt(1 / eigen_values[3]) * math.sin(angle_3)
                s = Q @ np.array([y0, y1, y2, y3]).transpose()
                # print(s.transpose() @ MATRIX_P @ s)
                initial_condition_list.append([s[0], s[1], s[2], s[3], False])

    print(f"Generating {len(initial_condition_list)} conditions for training ...")

    return initial_condition_list
