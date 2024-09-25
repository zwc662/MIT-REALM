import einops as ei
import jax
import jax.nn as jnn
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Float

from pncbf.dyn.dyn_types import BState, Control, Disturb, HFloat, LFloat, PolObs, State, VObs
from pncbf.dyn.task import Task
from pncbf.plotting.phase2d_utils import plot_x_bounds, plot_x_goal, plot_y_bounds
from pncbf.plotting.plotstyle import PlotStyle
from pncbf.utils.constraint_utils import BoxConstraint
from pncbf.utils.jax_types import Arr, BoolScalar
from pncbf.utils.jax_utils import jax_vmap
from pncbf.utils.sampling_utils import get_mesh_np


class BoxTwoDInt(Task):

    NX = 2
    NU = 1
    ND = 0

    P, V = range(NX)
    (A,) = range(NU)

    GOAL_X, GOAL_R = 0.75, 0.02
    GOAL_XMIN, GOAL_XMAX = GOAL_X - GOAL_R, GOAL_X + GOAL_R

    @property
    def nx(self) -> int:
        return self.NX

    @property
    def nu(self) -> int:
        return self.NU

    @property
    def nd(self) -> int:
        return self.ND

    @property
    def nl(self) -> int:
        return len(self.l_labels)

    @property
    def nh(self) -> int:
        return len(self.h_labels)

    @property
    def n_Vobs(self) -> int:
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
        return [r"$p_{lb}$", r"$p_{ub}$", r"$p_{lb}$", r"$v_{ub}$"]

    @property
    def l_scale(self) -> LFloat:
        return np.array([30.0])

    def l_components(self, state: State) -> LFloat:
        self.chk_x(state)
        p, v = state
        dist = jnn.relu(jnp.abs(p - self.GOAL_X) - self.GOAL_R)
        # Saturate it.
        l_dist = jnp.tanh(dist)
        return jnp.stack([l_dist]) / self.l_scale

    @property
    def h_max(self) -> float:
        return 5.0

    def h_components(self, state: State) -> HFloat:
        self.chk_x(state)
        p, v = state
        h_p_lb, h_p_ub = BoxConstraint(-1.0, 1.0).to_array(p)
        h_v_lb, h_v_ub = BoxConstraint(-1.0, 1.0).to_array(v)
        return jnp.array([h_p_lb, h_p_ub, h_v_lb, h_v_ub]).clip(max=self.h_max)

    def get_obs(self, state: State) -> tuple[VObs, PolObs]:
        return state, state

    @property
    def dt(self):
        return 0.1

    def step(self, state: State, control: Control, disturb: Disturb = None) -> State:
        dt = self.dt
        p, v = state
        (a,) = control
        p_new = p + v * dt + 0.5 * a * dt**2
        v_new = v + a * dt
        return self.chk_x(jnp.array([p_new, v_new]))

    def train_bounds(self) -> Float[Arr, "2 nx"]:
        return np.array([(-1.5, 1.5), (-1.5, 1.5)]).T

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
        return np.array([[-0.75, 0.75]])

    def in_ci_approx(self, state: State) -> BoolScalar:
        x, v = state
        in_right = x <= 1 - jnp.maximum(v, 0) ** 2 / 2
        in_left = x >= jnp.minimum(v, 0) ** 2 / 2 - 1
        within_v = jnp.abs(v) <= 1
        return in_left & in_right & within_v

    def nominal_val_state(self) -> State:
        return np.array([0.0, 0.0])

    def _phase2d_setups(self) -> list[Task.Phase2DSetup]:
        return [Task.Phase2DSetup("phase", self.plot_phase, Task.mk_get2d([self.P, self.V]))]

    def plot_phase(self, ax: plt.Axes):
        """(Position, Velocity) plot."""
        PLOT_XMIN, PLOT_XMAX = -1.5, 1.5
        PLOT_YMIN, PLOT_YMAX = -1.5, 1.5
        ax.set(xlim=(PLOT_XMIN, PLOT_XMAX), ylim=(PLOT_YMIN, PLOT_YMAX))

        # 1: Plot CI clockwise.
        all_xs, all_vs = [], []
        vs = np.linspace(-1.0, 1.0)
        xs = np.minimum(vs, 0.0) ** 2 / 2 - 1
        all_xs += [xs]
        all_vs += [vs]

        xs = np.linspace(-1.0, 0.5)[1:-1]
        vs = np.full_like(xs, 1.0)
        all_xs += [xs]
        all_vs += [vs]

        vs = np.linspace(-1.0, 1.0)[::-1]
        xs = 1 - np.maximum(vs, 0.0) ** 2 / 2
        all_xs += [xs]
        all_vs += [vs]

        xs = np.linspace(-0.5, 1.0)[:-1][::-1]
        vs = np.full_like(xs, -1.0)
        all_xs += [xs]
        all_vs += [vs]

        xs, vs = np.concatenate(all_xs, axis=0), np.concatenate(all_vs, axis=0)
        ax.plot(xs, vs, **PlotStyle.ci_line)

        # 2: Plot the avoid set.
        plot_x_bounds(ax, (-1.0, 1.0), PlotStyle.obs_region)
        plot_y_bounds(ax, (-1.0, 1.0), PlotStyle.obs_region)

        # 3: Plot the goal.
        plot_x_goal(ax, (self.GOAL_XMIN, self.GOAL_XMAX), PlotStyle.goal_region)

        # 4: Plot the switching surface for the unconstrained.
        vs = np.linspace(PLOT_YMIN, PLOT_YMAX)
        xs = self.GOAL_XMIN - np.sign(vs) * vs**2 / 2
        ax.plot(xs, vs, **PlotStyle.switch_line)
        xs = self.GOAL_XMAX - np.sign(vs) * vs**2 / 2
        ax.plot(xs, vs, **PlotStyle.switch_line)
