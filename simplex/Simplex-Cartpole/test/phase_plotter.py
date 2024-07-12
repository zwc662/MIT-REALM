import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from numpy.linalg import inv
from numpy import linalg as LA
import matplotlib.pyplot as plt
from utils import *
from tqdm import tqdm
import copy


# from lib.agent.ddpg import DDPGAgent, DDPGParams
# from lib.env.cart_pole import CartpoleParams, Cartpole, states2observations

def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique), fontsize="7.3", loc='best')


def check_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        print(f"{dir_path} does not exist, creating...")


class PhasePlotter:
    def __init__(self, p_mat, phase_dir="plots/phases", trajectory_dir="plots/trajectories", simplex_enable=True):
        self.phase_dir = phase_dir
        self.trajectory_dir = trajectory_dir
        self.simplex_enable = simplex_enable
        self.p_mat = p_mat

        self.check_all_dir()

    def check_all_dir(self):
        check_dir(self.phase_dir)
        check_dir(self.trajectory_dir)

    def plot_safety_set(self, x_set=[-0.9, 0.9], theta_set=[-0.8, 0.8]):
        x_l, x_h = x_set
        th_l, th_h = theta_set

        # Safety Set
        plt.vlines(x=x_l, ymin=th_l, ymax=th_h, color='black', linewidth=2.5)
        plt.vlines(x=x_h, ymin=th_l, ymax=th_h, color='black', linewidth=2.5)
        plt.hlines(y=th_l, xmin=x_l, xmax=x_h, color='black', linewidth=2.5)
        plt.hlines(y=th_h, xmin=x_l, xmax=x_h, color='black', linewidth=2.5)

    def plot_envelope(self, epsilon, p_mat=None):
        if p_mat is None:
            p_mat = self.p_mat
        cP = p_mat

        tP = np.zeros((2, 2))
        vP = np.zeros((2, 2))

        # For velocity
        vP[0][0] = cP[1][1]
        vP[1][1] = cP[3][3]
        vP[0][1] = cP[1][3]
        vP[1][0] = cP[1][3]

        # For position
        tP[0][0] = cP[0][0]
        tP[1][1] = cP[2][2]
        tP[0][1] = cP[0][2]
        tP[1][0] = cP[0][2]

        wp, vp = LA.eig(tP)
        wp_eps, vp_eps = LA.eig(tP / epsilon)
        # wp, vp = LA.eig(vP)

        theta = np.linspace(-np.pi, np.pi, 1000)

        ty1 = (np.cos(theta)) / np.sqrt(wp[0])
        ty2 = (np.sin(theta)) / np.sqrt(wp[1])

        ty1_eps = (np.cos(theta)) / np.sqrt(wp_eps[0])
        ty2_eps = (np.sin(theta)) / np.sqrt(wp_eps[1])

        ty = np.stack((ty1, ty2))
        tQ = inv(vp.transpose())
        # tQ = vp.transpose()
        tx = np.matmul(tQ, ty)

        ty_eps = np.stack((ty1_eps, ty2_eps))
        tQ_eps = inv(vp_eps.transpose())
        tx_eps = np.matmul(tQ_eps, ty_eps)

        tx1 = np.array(tx[0]).flatten()
        tx2 = np.array(tx[1]).flatten()

        tx_eps1 = np.array(tx_eps[0]).flatten()
        tx_eps2 = np.array(tx_eps[1]).flatten()

        # Safety envelope
        plt.plot(tx1, tx2, linewidth=2, color='black')
        plt.plot(0, 0, 'k*', markersize=4, mew=0.6)  # global equilibrium (star)
        plt.plot(0, 0, 'ko-', markersize=7, mew=1, markerfacecolor='none')  # global equilibrium (circle)

        # HAC switch envelope
        # if self.simplex_enable:
        #     plt.plot(tx_eps1, tx_eps2, 'k--', linewidth=0.8, label=r"$\partial\Omega_{HAC}$")

        # HPC switch envelope
        # plt.plot(tx_hpc1, tx_hpc2, 'b--', linewidth=0.8, label=r"$\partial\Omega_{HPC}$")

    def plot_phase(self, trajectories, action_modes, eq_point, plot_eq=False):

        print(f"len traj: {len(trajectories)}")
        print(f"len action: {len(action_modes)}")
        assert len(trajectories) == len(action_modes)

        # eq points
        if plot_eq and eq_point is not None:
            print(f"eq point: {eq_point}")
            plt.plot(eq_point[0], eq_point[2], '*', color=[0.4660, 0.6740, 0.1880], markersize=8)

        for i in range(len(trajectories) - 1):
            if action_modes[i] == "model" or action_modes[i] == "residual":
                plt.plot(trajectories[i][0], trajectories[i][1], '.', color=[0, 0.4470, 0.7410],
                         markersize=2)  # model trajectory
            elif action_modes[i] == "simplex":
                plt.plot(trajectories[i][0], trajectories[i][1], 'r.', markersize=2)  # simplex trajectory
            else:
                raise RuntimeError("Unrecognized action mode")

        # Add label
        h1, = plt.plot(trajectories[-1][0], trajectories[-1][1], '.', color=[0, 0.4470, 0.7410], label="HPC",
                       markersize=2)
        if self.simplex_enable:
            h2, = plt.plot(trajectories[-1][0], trajectories[-1][1], 'r.', label="HAC", markersize=2)

        # plt.plot(trajectories[0][:], trajectories[1][:], 'r.', markersize=2)  # trajectory
        h3, = plt.plot(trajectories[0][0], trajectories[0][1], 'ko', markersize=6, mew=1.2)  # initial state
        h4, = plt.plot(trajectories[-1][0], trajectories[-1][1], 'kx', markersize=8, mew=1.2)  # end state

    def phase_portrait(self, trajectories, action_modes, x_set, theta_set, epsilon, idx, freq, eq_point,
                       plot_phase=True, plot_eq=False, p_mat=None):
        if plot_phase is False:
            return

        plt.close()
        plt.clf()
        p_mat = p_mat or self.p_mat

        # Phase
        self.plot_phase(trajectories=trajectories, action_modes=action_modes, eq_point=eq_point, plot_eq=plot_eq)

        # Safety envelope
        self.plot_envelope(p_mat=p_mat, epsilon=epsilon)

        # Safety set
        # self.plot_safety_set(x_set=x_set, theta_set=theta_set)

        # plt.title(f"Inverted Pendulum Phase ($f = {freq} Hz$)", fontsize=14)
        plt.xlabel('x (m)', fontsize=18)
        plt.ylabel('$\\theta$ (rad)', fontsize=18)
        plt.legend(loc="lower left", markerscale=4, handlelength=1.2, handletextpad=0.5, bbox_to_anchor=(0.05, 0.05))

        plt.savefig(f"{self.phase_dir}/phase{idx}.png")

        pass

    def plot_trajectory(self, x_set, theta_set, force_set, trajectories, action_modes, forces, safety_vals, freq, idx,
                        plot_traj=True):
        if plot_traj is False:
            return

        x_l, x_h = x_set[0], x_set[1]
        th_l, th_h = theta_set[0], theta_set[1]
        f_l, f_h = force_set[0], force_set[1]
        x_ticks = np.linspace(x_l, x_h, 5)
        th_ticks = np.linspace(th_l, th_h, 5)
        f_ticks = np.linspace(f_l, f_h, 5)

        plt.close()
        plt.clf()

        n1 = len(trajectories)
        n2 = len(action_modes)
        n3 = len(forces)
        n4 = len(safety_vals)

        assert n1 == n2
        assert n2 == n3
        assert n3 == n4
        fig, axes = plt.subplots(3, 2, figsize=(12, 6))  # Create a 2x2 subplot grid
        fig.suptitle(f'Inverted Pendulum Trajectories ($f = {freq} Hz$)', fontsize=11, ha='center', y=0.97)

        for i in range(n1 - 1):
            if action_modes[i] == "model" or action_modes[i] == "residual":
                # x
                axes[0, 0].plot([i, i + 1], [trajectories[i][0], trajectories[i + 1][0]], '-', label='HPC',
                                color=[0, 0.4470, 0.7410])
                # x_dot
                axes[0, 1].plot([i, i + 1], [trajectories[i][1], trajectories[i + 1][1]], '-', label='HPC',
                                color=[0, 0.4470, 0.7410])

                # theta
                axes[1, 0].plot([i, i + 1], [trajectories[i][2], trajectories[i + 1][2]], '-', label='HPC',
                                color=[0, 0.4470, 0.7410])

                # theta_dot
                axes[1, 1].plot([i, i + 1], [trajectories[i][3], trajectories[i + 1][3]], '-', label='HPC',
                                color=[0, 0.4470, 0.7410])

                # force
                axes[2, 0].plot([i, i + 1], [forces[i], forces[i + 1]], '-', label='HPC', color=[0, 0.4470, 0.7410])

                # safety values
                axes[2, 1].plot([i, i + 1], [safety_vals[i], safety_vals[i + 1]], '-', label='HPC',
                                color=[0, 0.4470, 0.7410])


            elif action_modes[i] == "simplex":

                # x
                axes[0, 0].plot([i, i + 1], [trajectories[i][0], trajectories[i + 1][0]], 'r-', label='HAC')

                # x_dot
                axes[0, 1].plot([i, i + 1], [trajectories[i][1], trajectories[i + 1][1]], 'r-', label='HAC')

                # theta
                axes[1, 0].plot([i, i + 1], [trajectories[i][2], trajectories[i + 1][2]], 'r-', label='HAC')

                # theta_dot
                axes[1, 1].plot([i, i + 1], [trajectories[i][3], trajectories[i + 1][3]], 'r-', label='HAC')

                # force
                axes[2, 0].plot([i, i + 1], [forces[i], forces[i + 1]], 'r-', label='HAC')

                # safety values
                axes[2, 1].plot([i, i + 1], [safety_vals[i], safety_vals[i + 1]], 'r-', label='HAC')

            else:
                raise RuntimeError("Unrecognized action mode")

        # Add label and title (x)
        axes[0, 0].set_yticks(x_ticks)
        axes[0, 0].set_ylabel("x (m)")
        legend_without_duplicate_labels(axes[0, 0])

        # Add label and title (x_dot)
        axes[0, 1].set_ylabel(r"$\dot{x}$ (m/s)")
        legend_without_duplicate_labels(axes[0, 1])

        # Add label and title (theta)
        axes[1, 0].set_yticks(th_ticks)
        axes[1, 0].set_ylabel('$\\theta$ (rad)')
        legend_without_duplicate_labels(axes[1, 0])

        # Add label and title (theta_dot)
        axes[1, 1].set_ylabel(r'$\dot{\theta}$ (rad/s)')
        legend_without_duplicate_labels(axes[1, 1])

        # Add label and title (force)
        axes[2, 0].set_yticks(f_ticks)
        axes[2, 0].set_ylabel("force (N)")
        legend_without_duplicate_labels(axes[2, 0])

        # Add label and title (safety values)
        axes[2, 1].set_ylabel("safety value")
        legend_without_duplicate_labels(axes[2, 1])

        plt.tight_layout()  # Adjust spacing between subplots
        plt.savefig(f'{self.trajectory_dir}/trajectory{idx}.png', dpi=150)
