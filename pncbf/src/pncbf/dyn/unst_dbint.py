import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Float

from pncbf.dyn.dyn_types import Control, HFloat, PolObs, State, VObs
from pncbf.dyn.task import Task
from pncbf.plotting.phase2d_utils import plot_x_bounds, plot_y_bounds
from pncbf.plotting.plotstyle import PlotStyle
from pncbf.utils.costconstr_utils import poly4_clip_max_flat, poly4_softclip_flat
from pncbf.utils.jax_types import Arr


class UnstDbInt(Task):
    NX = 2
    NU = 1

    X1, X2 = range(NX)

    DT = 0.04

    def __init__(self):
        self._dt = UnstDbInt.DT

    @property
    def dt(self):
        return self._dt

    @property
    def n_Vobs(self) -> int:
        return 2

    @property
    def x_labels(self) -> list[str]:
        return [r"$x_1$", r"$x_2$"]

    @property
    def u_labels(self) -> list[str]:
        return [r"$u$"]

    @property
    def h_labels(self) -> list[str]:
        return [r"$1_l$", r"$1_u$", r"$2_l$", r"$2_u$"]

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
        x1, x2 = self.chk_x(state)
        h_x1_lb = -(x1 + 1.0)
        h_x1_ub = -(1.0 - x1)
        h_x2_lb = -(x2 + 1.0)
        h_x2_ub = -(1.0 - x2)
        hs = jnp.array([h_x1_lb, h_x1_ub, h_x2_lb, h_x2_ub])
        # h <= 1
        hs = poly4_clip_max_flat(hs)
        # softclip the minimum.
        hs = -poly4_softclip_flat(-hs, m=0.3)
        # hardclip the minimum.
        hs = -poly4_clip_max_flat(-hs, max_val=-self.h_min)
        return hs

    def get_obs(self, state: State) -> tuple[VObs, PolObs]:
        self.chk_x(state)
        obs = state
        return obs, obs

    def f(self, state: State) -> State:
        x1, x2 = self.chk_x(state)
        f = jnp.array([x1, -x1 + 0.5 * x2 + x2**3])
        # If x2 is too large, then just set it to zero to prevent escape.
        in_bounds = jnp.abs(x2) <= 2.0
        f = jnp.where(in_bounds, f, 0.0)
        return f

    def G(self, state: State) -> State:
        self.chk_x(state)
        G = np.zeros((self.NX, self.NU))
        G[0, 0] = 1.0
        return G

    def has_eq_state(self) -> bool:
        return True

    def eq_state(self) -> State:
        return np.zeros(self.NX)

    def nominal_val_state(self) -> State:
        return np.zeros(self.NX)

    def train_bounds(self) -> Float[Arr, "2 nx"]:
        return np.array([(-1.5, 1.5), (-1.5, 1.5)]).T

    def contour_bounds(self) -> Float[Arr, "2 nx"]:
        return self.train_bounds()

    def plot_bounds(self) -> Float[Arr, "2 nx"]:
        return self.train_bounds()

    def _phase2d_setups(self) -> list[Task.Phase2DSetup]:
        return [Task.Phase2DSetup("phase", self.plot_phase, Task.mk_get2d([self.X1, self.X2]))]

    def plot_phase(self, ax: plt.Axes):
        plot_bounds = self.plot_bounds()
        PLOT_X_MIN, PLOT_X_MAX = plot_bounds[:, self.X1]
        PLOT_Y_MIN, PLOT_Y_MAX = plot_bounds[:, self.X2]
        ax.set(xlim=(PLOT_X_MIN, PLOT_X_MAX), ylim=(PLOT_Y_MIN, PLOT_Y_MAX))

        # Plot the avoid set.
        plot_x_bounds(ax, (-1, 1), PlotStyle.obs_region)
        plot_y_bounds(ax, (-1, 1), PlotStyle.obs_region)
        ax.set(xlabel=r"$x_1$", ylabel=r"$x_2$")

    def nom_pol_zero(self, state: State) -> Control:
        return np.zeros(self.NU)
