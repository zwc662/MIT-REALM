import functools as ft

import einops as ei
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from attrs import define
from jaxtyping import Float

from clfrl.dyn.dyn_types import BState, Control, Disturb, HFloat, LFloat, PolObs, State, TState, VObs
from clfrl.dyn.odeint import rk4, tsit5
from clfrl.dyn.task import Task
from clfrl.plotting.phase2d_utils import plot_x_bounds
from clfrl.plotting.plotstyle import PlotStyle
from clfrl.utils.jax_types import Arr, BoolScalar, TFloat
from clfrl.utils.jax_utils import jax_vmap
from clfrl.utils.none import get_or
from clfrl.utils.sampling_utils import get_mesh_np


class Pend(Task):
    NX = 2
    NU = 1
    ND = 1

    TH, W = range(NX)
    (TAU,) = range(NU)

    @define
    class Params:
        m: float = 1.0
        l: float = 0.25
        b: float = 0.01
        g: float = 9.81

    def __init__(self, params=Params()):
        self.params = params
        self.umax = 10.0

        self._dt = 0.1

    @property
    def nd(self) -> int:
        return self.ND

    @property
    def n_Vobs(self) -> int:
        return 3

    @property
    def x_labels(self) -> list[str]:
        return [r"$\theta$", r"$\omega$"]

    @property
    def u_labels(self) -> list[str]:
        return [r"$\tau$"]

    @property
    def l_labels(self) -> list[str]:
        return ["dist"]

    @property
    def h_labels(self) -> list[str]:
        return []

    @property
    def l_scale(self) -> LFloat:
        return np.array([40.0])

    def at_goal(self, state: State) -> BoolScalar:
        theta, w = self.chk_x(state)
        cos_theta = jnp.cos(theta)
        return jnp.abs(cos_theta + 1) < 1e-3

    def l_cts(self, state: State) -> LFloat:
        theta, w = self.chk_x(state)
        cos_theta = jnp.cos(theta)
        dist_cost = jnp.abs(cos_theta + 1)
        return dist_cost

    def l_components(self, state: State) -> LFloat:
        theta, w = self.chk_x(state)
        cos_theta = jnp.cos(theta)
        dist_cost = 0.5 * jnp.abs(cos_theta + 1) ** 2
        at_goal = self.at_goal(state)
        cost = jnp.array([dist_cost]) / self.l_scale
        cost = jnp.where(at_goal, 0.0, 0.05) + 0.5 * cost
        return cost

    @property
    def h_max(self) -> float:
        return 1.0

    def h_components(self, state: State) -> HFloat:
        return jnp.zeros(0)

    def is_stable(self, T_state: TState) -> BoolScalar:
        # If the position and velocity haven't changed significantly in the last 5 steps.
        return jnp.array(True)
        # max_state_diff = jnp.abs(T_state[-5:] - T_state[-5]).max(axis=0)
        # self.chk_x(max_state_diff)
        # pos_same = max_state_diff[0] < 0.02
        # vel_same = max_state_diff[1] < 0.02
        # return pos_same & vel_same

    def get_obs(self, state: State) -> tuple[VObs, PolObs]:
        # Prevent obs from exploding if the state explodes.
        theta, w = self.chk_x(state)
        obs = jnp.array([jnp.sin(theta), jnp.cos(theta), w])
        obs_scale = np.array([1.0, 1.0, 15.0])
        obs = obs / obs_scale
        return obs, obs

    @property
    def dt(self):
        return self._dt

    def f(self, state: State) -> State:
        p = self.params
        theta, w = self.chk_x(state)
        sin = jnp.sin(theta)
        d2theta = (-p.b / (p.m * p.l)) * w - (p.g / p.l) * sin
        return jnp.array([w, d2theta])

    def G(self, state: State):
        GT = np.array([[0, 1 / self.params.m]])
        G = GT.T
        return G * self.umax

    def step(self, state: State, control: Control, disturb: Disturb = None) -> State:
        xdot_with_u = ft.partial(self.xdot, control=control)
        return rk4(self.dt, xdot_with_u, state)

    def step_plot(
        self, state: State, control: Control, disturb: Disturb = None, dt: float = None
    ) -> tuple[TState, TFloat]:
        xdot_with_u = ft.partial(self.xdot, control=control)
        dt = get_or(dt, self.dt)
        return tsit5(dt, 4, xdot_with_u, state), np.linspace(0, dt, num=5)

    def train_bounds(self) -> Float[Arr, "2 nx"]:
        return np.array([(-np.pi, np.pi), (-20.0, 20.0)]).T

    def contour_bounds(self) -> Float[Arr, "2 nx"]:
        return np.array([(-1.3 * np.pi, 1.3 * np.pi), (-18.0, 18.0)]).T

    def get_plot_x0(self) -> BState:
        with jax.ensure_compile_time_eval():
            n_pts, idxs = 12, (0, 1)
            bb_Xs, bb_Ys, bb_x0 = get_mesh_np(self.train_bounds(), idxs, n_pts, n_pts, self.nominal_val_state())
            b_x0 = ei.rearrange(bb_x0, "nys nxs nx -> (nys nxs) nx")

            # Add some random noise to the positions to make it a bit less grid-like.
            rng = np.random.default_rng(seed=123124)
            b_pos_noise = 0.05 * rng.standard_normal((b_x0.shape[0], 2))
            b_x0[:, :2] += b_pos_noise

            # Only keep the ones that are inside.
            b_in_ci = jax_vmap(self.in_ci_approx)(b_x0)
            b_x0 = b_x0[b_in_ci]

        return b_x0

    def get_plot_rng_x0(self) -> BState:
        return np.array([[0.1, 0.0]])

    def in_ci_approx(self, state: State) -> BoolScalar:
        return jnp.array(True)

    def nominal_val_state(self) -> State:
        return np.array([np.pi, 0.0])

    def _phase2d_setups(self) -> list[Task.Phase2DSetup]:
        return [Task.Phase2DSetup("phase", self.plot_phase, Task.mk_get2d([self.TH, self.W]))]

    def plot_phase(self, ax: plt.Axes):
        """(Position, Velocity) plot."""
        PLOT_XMIN, PLOT_XMAX = -1.7 * np.pi, 1.7 * np.pi
        PLOT_YMIN, PLOT_YMAX = -22.0, 22.0
        ax.set(xlim=(PLOT_XMIN, PLOT_XMAX), ylim=(PLOT_YMIN, PLOT_YMAX))

        # Horizontal and vertical line at the goal.
        ax.axvline(-np.pi)
        ax.axvline(np.pi)
        ax.axhline(0.0)
