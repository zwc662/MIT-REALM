from lib.agent.ddpg import DDPGParams, DDPGAgent
from lib.agent.network import TaylorParams
from lib.env.cart_pole import CartpoleParams, Cartpole, states2observations, get_init_condition, MATRIX_P
from lib.env.cart_pole import observations2states
from lib.logger.logger import LoggerParams, Logger, plot_trajectory
import matplotlib.pyplot as plt
from lib.utils import ReplayMemory
from numpy import linalg as LA
from numpy.linalg import inv
import numpy as np
import time
import copy
import os

np.set_printoptions(suppress=True)


class Params:
    def __init__(self):
        self.cartpole_params = CartpoleParams()
        self.agent_params = DDPGParams()
        self.logger_params = LoggerParams()
        self.taylor_params = TaylorParams()


class CartpoleDDPG:
    def __init__(self, params: Params):
        self.params = params
        self.cartpole = Cartpole(self.params.cartpole_params)

        self.shape_observations = self.cartpole.states_observations_dim
        self.shape_action = self.cartpole.action_dim
        self.replay_mem = ReplayMemory(self.params.agent_params.total_training_steps)

        if self.params.cartpole_params.observe_reference_states:
            self.shape_observations += self.cartpole.states_observations_refer_dim

        self.agent = DDPGAgent(self.params.agent_params,
                               self.params.taylor_params,
                               shape_observations=self.shape_observations,
                               shape_action=self.shape_action,
                               model_path=self.params.agent_params.model_path,
                               mode=self.params.logger_params.mode)

        self.logger = Logger(self.params.logger_params)
        self.failed_times = 0

    def interaction_step(self, mode=None):

        current_states = copy.deepcopy(self.cartpole.states)

        observations, _ = states2observations(current_states)

        drl_action = self.agent.get_action(observations, mode)

        next_states, nominal_drl_action = self.cartpole.step(drl_action,
                                                             action_mode=self.params.agent_params.action_mode)

        if self.params.cartpole_params.update_reference_model:
            self.cartpole.refer_step()

        observations_next, failed = states2observations(next_states)

        r, distance_score = self.cartpole.reward_fcn(current_states, nominal_drl_action, next_states)

        return observations, nominal_drl_action, observations_next, failed, r, distance_score

    def evaluation(self, reset_states=None, mode=None):

        if self.params.cartpole_params.random_reset_eval:
            self.cartpole.random_reset(mode=mode)
        else:
            self.cartpole.reset(reset_states)

        reward_list = []
        distance_score_list = []
        failed = False
        trajectory_tensor = []

        if mode == 'test':
            self.cartpole.force_list.clear()
            self.cartpole.state_list.clear()
            self.cartpole.safety_val_list.clear()
            self.cartpole.eq_point = None
            # self.agent.params.action_mode = 'residual'
            state_traj = []
            state_four = []
            state_four.append(np.array(self.cartpole.states[0:4]))
            state_mode = [self.params.agent_params.action_mode]
            state_traj.append(np.array([self.cartpole.states[0], self.cartpole.states[2]]))
            self.cartpole.state_list.append(self.cartpole.states[:4])
            self.cartpole.force_list.append(0)
            s = np.array([self.cartpole.states[0], self.cartpole.states[2]])

            sv = self.cartpole.safety_value(states=np.array(self.cartpole.states[0:4]), p_mat=self.cartpole.p_mat)
            self.cartpole.safety_val_list.append(sv)

        dwell_steps = 0
        for step in range(self.params.agent_params.max_episode_steps):

            if mode == 'test':
                states = copy.deepcopy(self.cartpole.states[:4])
                curr_state = np.array(states)
                safety_value = curr_state @ MATRIX_P @ curr_state.transpose()
                tensor = [safety_value] + states
                trajectory_tensor.append(tensor)
                state_four.append(curr_state)
                state_traj.append(np.array([states[0], states[2]]))
                state_mode.append(self.params.agent_params.action_mode)

                # Inside simplex-epsilon
                if self.cartpole.safety_value(states=curr_state,
                                              p_mat=MATRIX_P) <= self.params.cartpole_params.simplex_epsilon:

                    # HAC Dwell-Time
                    if self.cartpole.last_action_mode == "simplex":

                        if dwell_steps < self.cartpole.params.dwell_steps:
                            print(f"current dwell steps: {dwell_steps}")
                            dwell_steps += 1

                        # Switch back to HPC (If Simplex enabled)
                        elif self.cartpole.params.continue_learn:
                            # self.params.agent_params.action_mode = "model"
                            self.params.agent_params.action_mode = "residual"
                            print(f"Simplex control switch back to {self.params.agent_params.action_mode} control")

                # Outside simplex-epsilon
                if self.cartpole.safety_value(states=curr_state,
                                              p_mat=MATRIX_P) > self.params.cartpole_params.simplex_epsilon:
                    # print(f"theta: {theta}, theta_dot: {theta_dot}")
                    # ss = np.array([x, x_dot, theta, theta_dot])
                    x = curr_state[0]
                    x_dot = curr_state[1]
                    theta = curr_state[2]
                    theta_dot = curr_state[3]

                    # Switch from HPC to HAC (if enabled)
                    if (self.cartpole.last_action_mode in ["residual", "model"]
                            and self.cartpole.params.simplex_enable):
                        # self.cartpole.eq_point = np.array(
                        #     [x * self.params.cartpole_params.chi, 0, 0, 0])  # record eq point (patch center)
                        self.cartpole.eq_point = (np.array([x, x_dot, theta, theta_dot])
                                                  * self.params.cartpole_params.chi)  # (patch center)

                        sbar_star = np.array([x, x_dot, theta, theta_dot]) * self.params.cartpole_params.chi
                        # sbar_star = self.cartpole.eq_point

                        Ac, Bc = self.cartpole.get_A_B_mat_by_state(state=sbar_star)
                        print(f"State Approximation:\nA: {Ac}\nB: {Bc}")

                        s = time.time()
                        self.cartpole.calculate_F(Ac=Ac, Bc=Bc)
                        print(f"F_hat is: {self.cartpole.F_hat}")
                        e = time.time()
                        print(f"LMI time duration: {e - s}")

                        # Continuous to discrete form
                        Ak, Bk = self.cartpole.get_discrete_A_and_B(Ac, Bc)
                        self.cartpole.Ak = Ak
                        self.cartpole.Bk = Bk

                        # Switch to HAC Simplex
                        self.params.agent_params.action_mode = "simplex"

                else:
                    # self.params.agent_params.action_mode = "model"
                    pass

            observations, action, observations_next, failed, r, distance_score = \
                self.interaction_step(mode='eval')

            if self.params.logger_params.visualize_eval:
                self.cartpole.render()

            reward_list.append(r)
            distance_score_list.append(distance_score)

            if self.params.cartpole_params.use_termination and failed:
                break

        mean_reward = np.mean(reward_list)
        mean_distance_score = np.mean(distance_score_list)

        if mode == 'test':
            plot_trajectory(trajectory_tensor)

            eq = self.cartpole.eq_point
            x_h = self.cartpole.params.x_threshold
            f_h = self.cartpole.params.force_threshold
            th_h = self.cartpole.params.theta_threshold
            x_l = -x_h
            th_l = -th_h
            f_l = -f_h
            self.cartpole.plotter.phase_portrait(
                trajectories=state_traj,
                action_modes=state_mode,
                x_set=[x_l, x_h],
                theta_set=[th_l, th_h],
                epsilon=self.params.cartpole_params.simplex_epsilon,
                freq=self.cartpole.params.simulation_frequency,
                eq_point=eq,
                plot_phase=self.params.cartpole_params.phase_plot,
                plot_eq=True,
                idx="_eval"
            )

            self.cartpole.plotter.plot_trajectory(
                x_set=[x_l, x_h],
                theta_set=[th_l, th_h],
                force_set=[f_l, f_h],
                trajectories=self.cartpole.state_list,
                safety_vals=self.cartpole.safety_val_list,
                action_modes=state_mode,
                forces=self.cartpole.force_list,
                plot_traj=self.cartpole.params.trajectory_plot,
                freq=self.cartpole.params.simulation_frequency,
                idx="_eval"
            )

        # file_name = "safe_learn_trajectory1.txt"
        # cnt = np.array(state_four)
        # np.savetxt(file_name, cnt)

        self.cartpole.close()

        return mean_reward, mean_distance_score, failed

    def train(self):
        ep = 0
        global_steps = 0
        best_dsas = 0.0  # Best distance score and survived
        moving_average_dsas = 0.0
        initial_condition = get_init_condition(n_points_per_dim=self.params.cartpole_params.n_points_per_dim)

        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # print(f"initial_condition: {initial_condition}")
        # print(f"initial_condition len: {len(initial_condition)}")
        # print(f"iteration_times: {self.params.agent_params.iteration_times}")
        # time.sleep(123)
        # here we use episode-based counter, since we focus more on initial conditions

        for _ in range(self.params.agent_params.iteration_times):

            for i, cond in enumerate(initial_condition):

                # cond = [-0.024478718101496655, -0.5511401911926881, 0.43607751272052686, -0.25180280548180833, False]
                if self.params.cartpole_params.random_reset_train:
                    self.cartpole.random_reset(mode='train')
                else:
                    self.cartpole.reset(reset_states=cond)

                # Parameters Reset for Plot traj (position)
                dwell_steps = 0
                self.cartpole.force_list.clear()
                self.cartpole.state_list.clear()
                self.cartpole.safety_val_list.clear()
                self.cartpole.eq_point = None
                self.agent.params.action_mode = 'residual'
                state_traj = []
                state_mode = [self.params.agent_params.action_mode]
                state_traj.append(np.array([self.cartpole.states[0], self.cartpole.states[2]]))
                self.cartpole.state_list.append(self.cartpole.states[:4])
                self.cartpole.force_list.append(0)
                s = np.array([self.cartpole.states[0], self.cartpole.states[2]])
                sv = self.cartpole.safety_value(states=s, p_mat=self.cartpole.pP)
                self.cartpole.safety_val_list.append(sv)

                # state_traj[1].append(self.cartpole.states[2])

                print(f"Training at {i} init_cond:{self.cartpole.states[:4]}")

                reward_list = []
                distance_score_list = []
                critic_loss_list = []
                failed = False

                ep_steps = 0
                for step in range(self.params.agent_params.max_episode_steps):

                    observations, action, observations_next, failed, r, distance_score = \
                        self.interaction_step(mode='train')

                    self.replay_mem.add((observations, action, r, observations_next, failed))

                    reward_list.append(r)

                    distance_score_list.append(distance_score)

                    if self.replay_mem.get_size() > self.params.agent_params.experience_prefill_size:
                        minibatch = self.replay_mem.sample(self.params.agent_params.batch_size)
                        critic_loss = self.agent.optimize(minibatch)
                    else:
                        critic_loss = 100

                    critic_loss_list.append(critic_loss)
                    global_steps += 1
                    ep_steps += 1

                    # Append the state point
                    x, x_dot, theta, theta_dot = observations2states(observations, failed)[:4]
                    curr_state = np.array([x, x_dot, theta, theta_dot])
                    s = np.array([x, theta])
                    pP = self.cartpole.pP

                    # state_traj[0].append(x)
                    # state_traj[1].append(theta)
                    self.cartpole.state_list.append(self.cartpole.states[:4])
                    state_traj.append(np.array([x, theta]))
                    state_mode.append(self.params.agent_params.action_mode)
                    sv = self.cartpole.safety_value(states=curr_state, p_mat=MATRIX_P)
                    self.cartpole.safety_val_list.append(sv)

                    # Inside simplex-epsilon
                    if self.cartpole.safety_value(states=curr_state,
                                                  p_mat=MATRIX_P) <= self.params.cartpole_params.simplex_epsilon:

                        # HAC Dwell-Time
                        if self.cartpole.last_action_mode == "simplex":

                            if dwell_steps < self.cartpole.params.dwell_steps:
                                print(f"current dwell steps: {dwell_steps}")
                                dwell_steps += 1

                            # Switch back to HPC (If Simplex enabled)
                            elif self.cartpole.params.continue_learn:
                                # self.params.agent_params.action_mode = "model"
                                self.params.agent_params.action_mode = "residual"
                                print(f"Simplex control switch back to {self.params.agent_params.action_mode} control")

                    # Outside simplex-epsilon
                    elif self.cartpole.safety_value(states=curr_state,
                                                  p_mat=MATRIX_P) > self.params.cartpole_params.simplex_epsilon:
                        # print(f"theta: {theta}, theta_dot: {theta_dot}")
                        ss = np.array([x, x_dot, theta, theta_dot])

                        # Switch from HPC to HAC (if enabled)
                        if (self.cartpole.last_action_mode in ["residual", "model"]
                                and self.cartpole.params.simplex_enable):

                            # self.cartpole.eq_point = np.array(
                            #     [x * self.params.cartpole_params.chi, 0, 0, 0])  # record eq point (patch center)
                            self.cartpole.eq_point = (np.array([x, x_dot, theta, theta_dot])
                                                      * self.params.cartpole_params.chi)  # (patch center)

                            sbar_star = np.array([x, x_dot, theta, theta_dot]) * self.params.cartpole_params.chi
                            # sbar_star = self.cartpole.eq_point

                            Ac, Bc = self.cartpole.get_A_B_mat_by_state(state=sbar_star)
                            print(f"State Approximation:\nA: {Ac}\nB: {Bc}")

                            s = time.time()
                            self.cartpole.calculate_F(Ac=Ac, Bc=Bc)
                            print(f"F_hat is: {self.cartpole.F_hat}")
                            e = time.time()
                            print(f"LMI time duration: {e - s}")

                            # Continuous to discrete form
                            Ak, Bk = self.cartpole.get_discrete_A_and_B(Ac, Bc)
                            self.cartpole.Ak = Ak
                            self.cartpole.Bk = Bk

                            # Record control information
                            info_dir = "plots/info"
                            if not os.path.exists(info_dir):
                                os.makedirs(info_dir)
                            f = open(f"{info_dir}/{i}.txt", "w")
                            cnt = (f"s is:\n{ss}\n"
                                   f"F_hat is:\n{self.cartpole.F_hat}\n"
                                   f"Ak is:\n{Ak}\n"
                                   f"Bk is:\n{Bk}\n"
                                   f"LMI duration: {e - s}\n")
                            f.write(cnt)
                            f.close()

                            # Switch to HAC Simplex
                            self.params.agent_params.action_mode = "simplex"

                    else:
                        # self.params.agent_params.action_mode = "model"
                        pass

                    if failed and self.params.cartpole_params.use_termination:
                        print(f"step is: {step}")

                        # self.params.agent_params.action_mode = "model"
                        break

                    # if step == 100:
                    #     break
                print(f"Plotting phase: {i}")
                eq = self.cartpole.eq_point
                x_h = self.cartpole.params.x_threshold
                f_h = self.cartpole.params.force_threshold
                th_h = self.cartpole.params.theta_threshold
                x_l = -x_h
                th_l = -th_h
                f_l = -f_h
                self.cartpole.plotter.phase_portrait(
                    trajectories=state_traj,
                    action_modes=state_mode,
                    x_set=[x_l, x_h],
                    theta_set=[th_l, th_h],
                    epsilon=self.params.cartpole_params.simplex_epsilon,
                    freq=self.cartpole.params.simulation_frequency,
                    eq_point=eq,
                    plot_phase=self.params.cartpole_params.phase_plot,
                    plot_eq=True,
                    idx=i
                )

                self.cartpole.plotter.plot_trajectory(
                    x_set=[x_l, x_h],
                    theta_set=[th_l, th_h],
                    force_set=[f_l, f_h],
                    trajectories=self.cartpole.state_list,
                    safety_vals=self.cartpole.safety_val_list,
                    action_modes=state_mode,
                    forces=self.cartpole.force_list,
                    plot_traj=self.cartpole.params.trajectory_plot,
                    freq=self.cartpole.params.simulation_frequency,
                    idx=i
                )

                self.failed_times += 1 * failed
                mean_reward = np.mean(reward_list)
                mean_distance_score = np.mean(distance_score_list)
                mean_critic_loss = np.mean(critic_loss_list)

                self.logger.log_training_data(mean_reward, mean_distance_score, mean_critic_loss, failed, global_steps)

                print(f"average_reward: {mean_reward:.6}, "
                      f"distance_score: {mean_distance_score:.6},"
                      f"critic_loss: {mean_critic_loss:.6}, total_steps_ep: {ep_steps} ")

                if (ep + 1) % self.params.logger_params.evaluation_period == 0:
                    eval_mean_reward, eval_mean_distance_score, eval_failed = self.evaluation()
                    self.logger.log_evaluation_data(eval_mean_reward, eval_mean_distance_score, eval_failed,
                                                    global_steps)
                    moving_average_dsas = 0.95 * moving_average_dsas + 0.05 * eval_mean_distance_score
                    if moving_average_dsas > best_dsas:
                        self.agent.save_weights(self.logger.model_dir + '_best')
                        best_dsas = moving_average_dsas
                    # exit(0)
                ep += 1

                # print(f"global_steps is: {global_steps}")

                # here we want different approach to exit after the same interaction steps
                if global_steps > self.params.agent_params.total_training_steps:
                    self.agent.save_weights(self.logger.model_dir)
                    np.savetxt(f"logs/{self.params.logger_params.model_name}/failed_times.txt",
                               [self.failed_times, ep, self.failed_times / ep])
                    exit("Reach maximum steps, exit...")

                # print(f"ep is: {ep}")
                # if ep == 3:
                #     break

        self.agent.save_weights(self.logger.model_dir)
        np.savetxt(f"logs/{self.params.logger_params.model_name}/failed_times.txt",
                   [self.failed_times, ep, self.failed_times / ep])
        print("Total failed:", self.failed_times)

    def test(self):
        self.evaluation(mode='test', reset_states=self.params.cartpole_params.ini_states)
