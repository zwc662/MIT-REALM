import einops as ei
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax_f16.f16 import F16
from matplotlib import pyplot as plt
from typing_extensions import override

from efppo.task.dyn_types import Control, HFloat, LFloat, Obs, State
from efppo.task.f16_safeguarded import MORELLI_BOUNDS, F16Safeguarded
from efppo.task.task import Task, TaskState
from efppo.utils.angle_utils import rotx, roty, rotz
from efppo.utils.jax_types import BBFloat, BoolScalar, FloatScalar
from efppo.utils.jax_utils import box_constr_clipmax, box_constr_log1p, merge01, tree_add, tree_inner_product, tree_mac
from efppo.utils.plot_utils import plot_x_bounds, plot_y_bounds, plot_y_goal
from efppo.utils.rng import PRNGKey
from efppo.task.f16 import * 
 


class F16GCASFloorCeilDisturbed(F16GCASFloorCeil):
    def __init__(self):
        super(F16GCASFloorCeilDisturbed, self).__init__()
        ### Add a time variant angle disturbance ###
    
    def grid_contour(self) -> tuple[BBFloat, BBFloat, TaskState]:
        # Contour with ( x axis=θ, y axis=H )
        n_xs = 64
        n_ys = 64
        b_th = np.linspace(-1.2, 1.2, num=n_xs)
        b_h = np.linspace(200.0, 1000.0, num=n_ys)

        x0 = jnp.array(self.nominal_val_state())
        bb_x0 = ei.repeat(x0, "nx -> b1 b2 nx", b1=n_ys, b2=n_xs)

        bb_X, bb_Y = np.meshgrid(b_th, b_h)
        bb_x0 = bb_x0.at[:, :, F16Safeguarded.THETA].set(bb_X)
        bb_x0 = bb_x0.at[:, :, F16Safeguarded.H].set(bb_Y)

        return bb_X, bb_Y, bb_x0

    def setup_traj_plot(self, ax: plt.Axes):
        PLOT_XMIN, PLOT_XMAX = xlim = [-np.pi / 2, np.pi / 2]
        PLOT_YMIN, PLOT_YMAX = ylim = [-50.0, 1100.0]
        ax.set_facecolor("0.98")
        ax.set(xlim=xlim, ylim=ylim)
        ax.set(xlabel='theta', ylabel='h')

        halfpi = 0.99 * np.pi / 2

        # 2: Plot the avoid set
        obs_style = dict(facecolor="0.6", edgecolor="none", alpha=0.4, zorder=3, hatch="/")
        plot_x_bounds(ax, (-halfpi, halfpi), obs_style)
        plot_y_bounds(ax, (0.0, 1000.0), obs_style)

        # 3: Plot the goal set.
        goal_style = dict(facecolor="green", edgecolor="none", alpha=0.3, zorder=4.0)
        plot_y_goal(ax, (self.goal_h_min, self.goal_h_max), goal_style)

    def err_fn(self, state_var: State) -> State:
        return 0
        
    @override
    def step(self, state: State, control: Control) -> State:
        print(control.shape)
        control = self.discr_to_cts(control)
        self.chk_x(state)
        self.chk_u(control)

        # Integrate using RK4.
        xdot = lambda x: self._f16.xdot(x, control)
        state_new = ode4(xdot, self.dt, state)

        # Safeguard.
        is_valid = self._f16.is_state_valid(state_new)
        freeze_states = ~is_valid
        state_new = state_new.at[F16Safeguarded.FREEZE].set(freeze_states)

        ### Add time variant gaussian disturbances on angles ###
        print("step")
        state_new = self.get_dist_state(state_new)
        
        return state_new


    @override
    def get_obs(self, state: State) -> Obs:
        """Encode angles."""
        self.chk_x(state)

        # Learn position-invariant policy.
        state = state.at[F16Safeguarded.PN].set(0.0)

        # sin-cos encode angles.
        with jax.ensure_compile_time_eval():
            angle_idxs = np.array([F16.ALPHA, F16.BETA, F16.PHI, F16.THETA, F16.PSI])
            other_idxs = np.setdiff1d(np.arange(self.NX), angle_idxs)
            
        angles = state[angle_idxs]
 
        other = state[other_idxs]

        angles_enc = jnp.concatenate([jnp.cos(angles), jnp.sin(angles)], axis=0)
        state_enc = jnp.concatenate([other, angles_enc], axis=0)
        assert state_enc.shape == (self.NX + len(angle_idxs),)

        # Add extra features.
        vel_feats = compute_f16_vel_angles(state[: F16.NX])
        assert vel_feats.shape == (3,)
        state_enc = jnp.concatenate([state_enc, vel_feats], axis=0)

        # fmt: off
        obs_mean = np.array([3.4e+02, -1.7e-01, 2.9e-01, 1.0e-01, 0.0e+00, 1.0e-01, 3.1e+02, 12.0e+00, 3.0e-02, 2.1e-02, 2.4e00, 4.0e-01, 8.7e-01, 8.9e-01, 7.7e-01, 8.0e-01, 7.6e-01, 3.6e-01, -1.3e-02, -7.6e-03, -3.5e-01, -1.4e-03, 5.9e-01, -3.6e-03, 5.4e-01])
        obs_std = np.array([1.1e+02, 1.7e+00, 6.3e-01, 3.2e+00, 1.0e+00, 1.3e+02, 2.2e+02, 5.0e+00, 1.9e+00, 1.5e+00, 4.5e+00, 4.9e-01, 1.4e-01, 9.6e-02, 2.6e-01, 2.2e-01, 3.7e-01, 3.1e-01, 4.4e-01, 5.8e-01, 4.4e-01, 5.4e-01, 3.5e-01, 3.1e-01, 3.8e-01])
        # fmt: on

        state_enc = (state_enc - obs_mean) / obs_std

        # For better stability, clip the state_enc to not be too large.
        state_enc = jnp.clip(state_enc, -10.0, 10.0)

        return state_enc
    

    def get_dist_state(self, state: State) -> State: 
        theta_err = self.err_fn(state[F16Safeguarded.THETA])
        dist_state = state.at[F16Safeguarded.THETA].add(theta_err)
        return dist_state 
 