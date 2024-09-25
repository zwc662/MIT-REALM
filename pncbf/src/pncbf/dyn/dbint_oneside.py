import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import shapely
from jaxtyping import Float
from matplotlib.colors import to_rgba

from pncbf.dyn.dyn_types import Control, HFloat, PolObs, State, VObs
from pncbf.dyn.task import Task
from pncbf.plotting.phase2d_utils import plot_x_bounds, plot_y_bounds
from pncbf.plotting.plotstyle import PlotStyle
from pncbf.plotting.poly_to_patch import poly_to_patch
from pncbf.utils.costconstr_utils import poly4_clip_max_flat, poly4_softclip_flat
from pncbf.utils.jax_types import Arr, BoolScalar
from pncbf.utils.sampling_utils import get_mesh_np


class DbIntOneSide(Task):
    NX = 2
    NU = 1

    P, V = range(NX)
    (A,) = range(NU)

    DT = 0.1

    def __init__(self):
        self._dt = DbIntOneSide.DT

    @property
    def dt(self):
        return self._dt

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
    def h_labels(self) -> list[str]:
        return [r"$p_{lb}$", r"$p_{ub}$", r"$v_{lb}$"]

    @property
    def h_max(self) -> float:
        return 1.0

    @property
    def h_min(self) -> float:
        return -1.0

    @property
    def u_min(self) -> Control:
        return np.full(self.NU, -1.0)

    @property
    def u_max(self) -> Control:
        return np.full(self.NU, +1.0)

    def h_components(self, state: State) -> HFloat:
        p, v = self.chk_x(state)
        h_p_lb = -(p + 2.0)
        h_p_ub = p
        h_v_lb = -(v + 0.5)
        hs = jnp.array([h_p_lb, h_p_ub, h_v_lb])
        # h <= 1
        hs = poly4_clip_max_flat(hs)
        # softclip the minimum.
        hs = -poly4_softclip_flat(-hs, m=0.3)
        # hardclip the minimum.
        hs = -poly4_clip_max_flat(-hs, max_val=-self.h_min)
        return hs

    def assert_is_safe(self, state: State) -> BoolScalar:
        # If we are sufficiently far away from the wall.
        p, v = self.chk_x(state)

        is_assert_safe = (p <= -0.5) & (v <= 2.0)
        is_unsafe = self.h(state) > 0.0
        return is_assert_safe & (~is_unsafe)

        # is_assert_safe = (-1.5 <= p) & (p <= -0.5) & (-0.1 <= v) & (v <= 0.1)
        # return is_assert_safe

    def get_obs(self, state: State) -> tuple[VObs, PolObs]:
        self.chk_x(state)
        obs = state
        return obs, obs

    def f(self, state: State) -> State:
        self.chk_x(state)
        p, v = state
        return jnp.array([v, 0.0])

    def G(self, state: State) -> State:
        self.chk_x(state)
        GT = np.array([[0.0, 1.0]])
        G = GT.T
        return G

    def has_eq_state(self) -> bool:
        return True

    def eq_state(self) -> State:
        return np.zeros(self.NX)

    def nominal_val_state(self) -> State:
        return np.zeros(self.NX)

    def train_bounds(self) -> Float[Arr, "2 nx"]:
        return np.array([(-2.5, 0.5), (-1.0, 2.5)]).T

    def contour_bounds(self) -> Float[Arr, "2 nx"]:
        return self.train_bounds()

    def plot_bounds(self) -> Float[Arr, "2 nx"]:
        return self.train_bounds()

    def get_contour_paper_x0(self, n_pts: int = 80):
        bounds = np.array([(-2.1, 0.2), (0.0, 2.2)]).T
        idxs = (0, 1)
        bb_Xs, bb_Ys, bb_x0 = get_mesh_np(bounds, idxs, n_pts, n_pts, self.nominal_val_state())
        return bb_x0, bb_Xs, bb_Ys

    def _phase2d_setups(self) -> list[Task.Phase2DSetup]:
        return [Task.Phase2DSetup("phase", self.plot_phase, Task.mk_get2d([self.P, self.V]))]

    def get_ci_points(self):
        vs = np.linspace(0.0, 2.0)
        xs = -np.maximum(vs, 0.0) ** 2 / 2
        assert xs.ndim == vs.ndim == 1

        # Add bottom left. (-2.0, 0.0)
        xs = np.concatenate([xs, [-2.0]])
        vs = np.concatenate([vs, [0.0]])

        return np.stack([xs, vs], axis=1)

    def plot_phase(self, ax: plt.Axes):
        plot_bounds = self.train_bounds()
        PLOT_XMIN, PLOT_XMAX = plot_bounds[:, self.P]
        PLOT_YMIN, PLOT_YMAX = plot_bounds[:, self.V]
        ax.set(xlim=(PLOT_XMIN, PLOT_XMAX), ylim=(PLOT_YMIN, PLOT_YMAX))
        ax.set(xlabel=self.x_labels[0], ylabel=self.x_labels[1])

        # Plot the CI.
        ci_pts = self.get_ci_points()
        ax.plot(ci_pts[:, 0], ci_pts[:, 1], **PlotStyle.ci_line)

        # Plot the adversarial curve.
        vs = np.linspace(-2.0, 3.0)
        xs = np.maximum(vs, 0.0) ** 2 / 2 - 2
        ax.plot(xs, vs, **PlotStyle.switch_line)

        # Plot the avoid set.
        plot_x_bounds(ax, (None, 0), PlotStyle.obs_region)
        plot_y_bounds(ax, (-0.5, None), PlotStyle.obs_region)

    def plot_paper(self, ax: plt.Axes):
        PLOT_XMIN, PLOT_XMAX = -2.1, 0.2
        PLOT_YMIN, PLOT_YMAX = -0.02, 2.2
        ax.set(xlim=(PLOT_XMIN, PLOT_XMAX), ylim=(PLOT_YMIN, PLOT_YMAX))
        ax.set(xlabel=r"$p$", ylabel=r"$v$")

        PLOT_YMIN2 = PLOT_YMIN - 1.0

        outside_pts = [(PLOT_XMIN, PLOT_YMIN2), (PLOT_XMIN, PLOT_YMAX), (PLOT_XMAX, PLOT_YMAX), (PLOT_XMAX, PLOT_YMIN2)]
        outside = shapely.Polygon(outside_pts)

        # Plot the outside of the CI as a shaded region.
        ci_pts = self.get_ci_points()
        hole = shapely.Polygon(ci_pts)
        # patch = poly_to_patch(hole, facecolor="C0", edgecolor="none", alpha=0.5, zorder=3)
        # ax.add_patch(patch)

        ci_poly = outside.difference(hole)
        patch = poly_to_patch(ci_poly, facecolor="0.6", edgecolor="none", alpha=0.3, zorder=3)
        ax.add_patch(patch)
        hatch_color = "0.5"
        patch = poly_to_patch(
            ci_poly, facecolor="none", edgecolor=hatch_color, linewidth=0, zorder=3.1, hatch="."
        )
        ax.add_patch(patch)

        # Plot the obstacle.
        obs_style = dict(facecolor="0.45", edgecolor="none", alpha=0.55, zorder=3.2)
        plot_x_bounds(ax, (-2.0, 0.0), obs_style)
        obs_style = dict(facecolor="none", lw=1.0, edgecolor="0.4", alpha=0.8, zorder=3.4, hatch="/")
        plot_x_bounds(ax, (-2.0, 0.0), obs_style)

    def nom_pol_zero(self, state: State) -> Control:
        return np.zeros(self.NU)
