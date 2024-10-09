import functools as ft

import einops as ei
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import shapely
from attrs import define
from jaxtyping import Float

from clfrl.dyn.dyn_types import BState, Control, Disturb, HFloat, LFloat, PolObs, State, TState, VObs
from clfrl.dyn.odeint import rk4, tsit5
from clfrl.dyn.task import Task
from clfrl.networks.fourier_emb import PosEmbed
from clfrl.networks.mlp import mlp_partial
from clfrl.networks.pol_det import PolDet
from clfrl.networks.train_state import TrainState
from clfrl.plotting.phase2d_utils import plot_x_bounds
from clfrl.plotting.plotstyle import PlotStyle
from clfrl.plotting.poly_to_patch import poly_to_patch
from clfrl.utils.costconstr_utils import poly4_clip_max_flat
from clfrl.utils.jax_types import Arr, BoolScalar, TFloat
from clfrl.utils.jax_utils import jax_vmap
from clfrl.utils.none import get_or
from clfrl.utils.rng import PRNGKey
from clfrl.utils.sampling_utils import get_mesh_np


class DoubleIntWall(Task):
    NX = 2
    NU = 1
    ND = 1

    P, V = range(NX)
    (A,) = range(NU)

    DT = 0.1

    def __init__(self):
        self.umax = 1.0
        self._dt = DoubleIntWall.DT
        self.pos_wall = 1.0

    @property
    def nd(self) -> int:
        return self.ND

    @property
    def n_Vobs(self) -> int:
        # return 4
        return 2

    @property
    def x_labels(self) -> list[str]:
        return [r"$p$", r"$v$"]

    @property
    def u_labels(self) -> list[str]:
        return [r"$a$"]

    @property
    def l_labels(self) -> list[str]:
        return ["dist"]

    @property
    def h_labels(self) -> list[str]:
        return [r"$p_{l}$", r"$p_{u}$"]

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
        self.chk_x(state)
        p, v = state
        h_p_ub = p - 1.0
        h_p_lb = -(p + 1.0)

        # h <= 1
        hs = poly4_clip_max_flat(jnp.array([h_p_lb, h_p_ub]))
        return hs

    def is_stable(self, T_state: TState) -> BoolScalar:
        # If the position and velocity haven't changed significantly in the last 5 steps.
        return jnp.array(True)

    def get_obs(self, state: State) -> tuple[VObs, PolObs]:
        p, v = self.chk_x(state)
        # obs = jnp.array([p, v, 0.5 * jnp.maximum(v, 0) ** 2, 0.5 * jnp.minimum(v, 0) ** 2])
        obs = jnp.array([p, v])
        return obs, obs

    @property
    def dt(self):
        return self._dt

    def f(self, state: State) -> State:
        self.chk_x(state)
        p, v = state
        return jnp.array([v, 0.0])

    def G(self, state: State):
        self.chk_x(state)
        GT = np.array([[0.0, 1.0]])
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
        return np.array([(-2.5, 2.5), (-3.0, 3.0)]).T

    def contour_bounds(self) -> Float[Arr, "2 nx"]:
        return np.array([(-2.5, 2.5), (-3.0, 3.0)]).T

    def get_paper_ci_x0(self, n_pts: int = 80):
        with jax.ensure_compile_time_eval():
            bounds = np.array([(-1.1, 1.1), (-2.1, 2.1)]).T
            idxs = (0, 1)
            bb_Xs, bb_Ys, bb_x0 = get_mesh_np(bounds, idxs, n_pts, n_pts, self.nominal_val_state())
        return bb_x0, bb_Xs, bb_Ys

    def plot_bounds(self) -> Float[Arr, "2 nx"]:
        return np.array([(-1.5, 1.5), (-2.5, 2.5)]).T

    def get_plot_x0(self) -> BState:
        with jax.ensure_compile_time_eval():
            n_pts, idxs = 12, (0, 1)
            bb_Xs, bb_Ys, bb_x0 = get_mesh_np(self.plot_bounds(), idxs, n_pts, n_pts, self.nominal_val_state())
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
        return np.array([[-1.0, 0.0]])

    def in_ci_approx(self, state: State) -> BoolScalar:
        x, v = self.chk_x(state)
        in_right = x <= self.pos_wall - jnp.maximum(v, 0) ** 2 / 2
        in_left = x >= jnp.minimum(v, 0) ** 2 / 2 - self.pos_wall
        return in_left & in_right

    def nominal_val_state(self) -> State:
        return np.array([-1.0, 0.0])

    def has_eq_state(self) -> bool:
        return True

    def eq_state(self) -> State:
        return np.zeros(2)

    def _phase2d_setups(self) -> list[Task.Phase2DSetup]:
        return [Task.Phase2DSetup("phase", self.plot_phase, Task.mk_get2d([self.P, self.V]))]

    def nom_pol_osc(self, state: State):
        self.chk_x(state)
        K = np.array([[1.01, 0.2]])
        return np.clip(-K @ state, -1, 1.0)

    def nom_pol_rng(self, state: State, key: PRNGKey = jr.PRNGKey(58123)):
        self.chk_x(state)
        state_fake = np.ones(self.nx)
        mlp = mlp_partial([16, 16, 16])
        ff = ft.partial(PosEmbed, mlp, embed_dim=8, scale=1.2)
        pol_def = PolDet(ff, self.nu)
        random_pol = TrainState.create_from_def(key, pol_def, (state_fake,), tx=None)
        return random_pol.apply(state)

    def nom_pol_rng2(self, state: State):
        return self.nom_pol_rng(state, jr.PRNGKey(511247))

    def get_ci_points(self):
        all_xs, all_vs = [], []
        vs = np.linspace(-2.0, 2.0)
        xs = self.pos_wall - np.maximum(vs, 0.0) ** 2 / 2
        all_xs += [xs]
        all_vs += [vs]

        vs = np.linspace(-2.0, 2.0)[::-1]
        xs = np.minimum(vs, 0.0) ** 2 / 2 - self.pos_wall
        all_xs += [xs]
        all_vs += [vs]

        all_xs, all_vs = np.concatenate(all_xs), np.concatenate(all_vs)
        assert all_xs.ndim == all_vs.ndim == 1

        return np.stack([all_xs, all_vs], axis=1)

    def plot_phase(self, ax: plt.Axes):
        """(Position, Velocity) plot."""
        PLOT_XMIN, PLOT_XMAX = -1.5, 1.5
        PLOT_YMIN, PLOT_YMAX = -3.0, 3.0
        ax.set(xlim=(PLOT_XMIN, PLOT_XMAX), ylim=(PLOT_YMIN, PLOT_YMAX))
        ax.set(xlabel=self.x_labels[0], ylabel=self.x_labels[1])

        # Plot the CI.
        ci_pts = self.get_ci_points()
        ax.plot(ci_pts[:, 0], ci_pts[:, 1], **PlotStyle.ci_line)

        # Plot the obstacle.
        plot_x_bounds(ax, (-self.pos_wall, self.pos_wall), PlotStyle.obs_region)

    def plot_phase_paper(self, ax: plt.Axes):
        PLOT_XMIN, PLOT_XMAX = -1.25, 1.25
        PLOT_YMIN, PLOT_YMAX = -2.25, 2.25
        ax.set(xlim=(PLOT_XMIN, PLOT_XMAX), ylim=(PLOT_YMIN, PLOT_YMAX))
        ax.set(xlabel=self.x_labels[0], ylabel=self.x_labels[1])

        outside_pts = [(PLOT_XMIN, PLOT_YMIN), (PLOT_XMIN, PLOT_YMAX), (PLOT_XMAX, PLOT_YMAX), (PLOT_XMAX, PLOT_YMIN)]
        outside = shapely.Polygon(outside_pts)

        # Plot the outside of the CI as a shaded region.
        ci_pts = self.get_ci_points()
        hole = shapely.Polygon(ci_pts)

        ci_poly = outside.difference(hole)
        patch = poly_to_patch(ci_poly, facecolor="0.7", edgecolor="none", alpha=0.5, zorder=3)
        ax.add_patch(patch)

        # Plot the obstacle.
        obs_style = dict(facecolor="0.5", edgecolor="none", alpha=0.55, zorder=3.2)
        plot_x_bounds(ax, (-self.pos_wall, self.pos_wall), obs_style)
        obs_style = dict(facecolor="none", lw=1.0, edgecolor="0.4", alpha=0.8, zorder=3.4, hatch="/")
        plot_x_bounds(ax, (-self.pos_wall, self.pos_wall), obs_style)
