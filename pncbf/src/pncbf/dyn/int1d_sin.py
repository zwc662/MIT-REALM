import functools as ft

import einops as ei
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import shapely
from jaxtyping import Float

from pncbf.dyn.dyn_types import BState, Control, Disturb, HFloat, LFloat, PolObs, State, TState, VObs
from pncbf.dyn.odeint import rk4, tsit5
from pncbf.dyn.task import Task
from pncbf.networks.fourier_emb import PosEmbed
from pncbf.networks.mlp import mlp_partial
from pncbf.networks.pol_det import PolDet
from pncbf.networks.train_state import TrainState
from pncbf.plotting.phase2d_utils import plot_x_bounds
from pncbf.plotting.plotstyle import PlotStyle
from pncbf.plotting.poly_to_patch import poly_to_patch
from pncbf.utils.costconstr_utils import poly4_clip_max_flat
from pncbf.utils.jax_types import Arr, BFloat, BoolScalar, FloatScalar, TFloat
from pncbf.utils.jax_utils import jax_vmap, smoothmax
from pncbf.utils.none import get_or
from pncbf.utils.rng import PRNGKey
from pncbf.utils.sampling_utils import get_mesh_np


class Int1DSin(Task):
    NX = 1
    NU = 1

    (P,) = range(NX)
    (V,) = range(NU)

    DT = 0.1

    def __init__(self):
        self.umax = 1.0
        self._dt = Int1DSin.DT
        self.sin_coeff = 10.0

    @property
    def n_Vobs(self) -> int:
        return 1

    @property
    def x_labels(self) -> list[str]:
        return [r"$p$"]

    @property
    def u_labels(self) -> list[str]:
        return [r"$v$"]

    @property
    def h_labels(self) -> list[str]:
        return [r"$h$"]

    @property
    def h_max(self) -> float:
        return 1.0

    @property
    def h_min(self) -> float:
        return -1.0

    @property
    def max_ttc(self) -> float:
        return 5.0

    def bounds(self):
        return -1.1, 1.4

    def h_components(self, state: State) -> HFloat:
        self.chk_x(state)
        (x,) = state
        return jnp.sin(self.sin_coeff * x)

    def get_obs(self, state: State) -> tuple[VObs, PolObs]:
        (p,) = self.chk_x(state)
        # obs = jnp.array([p, v, 0.5 * jnp.maximum(v, 0) ** 2, 0.5 * jnp.minimum(v, 0) ** 2])
        obs = jnp.array([p])
        return obs, obs

    @property
    def dt(self):
        return self._dt

    def f(self, state: State) -> State:
        self.chk_x(state)
        return jnp.zeros(self.nx)

    def G(self, state: State):
        self.chk_x(state)
        return jnp.ones((self.nx, self.nu))

    def step(self, state: State, control: Control, disturb: Disturb = None) -> State:
        return state + self.dt * control

    def step_plot(
        self, state: State, control: Control, disturb: Disturb = None, dt: float = None
    ) -> tuple[TState, TFloat]:
        xdot_with_u = ft.partial(self.xdot, control=control)
        dt = get_or(dt, self.dt)
        return tsit5(dt, 4, xdot_with_u, state), np.linspace(0, dt, num=5)

    def train_bounds(self) -> Float[Arr, "2 nx"]:
        return np.array([self.bounds()]).T

    def contour_bounds(self) -> Float[Arr, "2 nx"]:
        return np.array([(self.bounds())]).T

    def get_paper_ci_x0(self, n_pts: int = 80):
        with jax.ensure_compile_time_eval():
            bounds = np.array([self.bounds()]).T
            idxs = (0, 1)
            bb_Xs, bb_Ys, bb_x0 = get_mesh_np(bounds, idxs, n_pts, n_pts, self.nominal_val_state())
        return bb_x0, bb_Xs, bb_Ys

    def get_paper_pi_x0(self, n_pts: int = 80):
        with jax.ensure_compile_time_eval():
            bounds = np.array([self.bounds()]).T
            idxs = (0, 1)
            bb_Xs, bb_Ys, bb_x0 = get_mesh_np(bounds, idxs, n_pts, n_pts, self.nominal_val_state())
        return bb_x0, bb_Xs, bb_Ys

    def plot_bounds(self) -> Float[Arr, "2 nx"]:
        return np.array([self.bounds()]).T

    def get_plot_rng_x0(self) -> BState:
        return np.array([[-0.01]])

    def nominal_val_state(self) -> State:
        return np.array([-0.01])

    def has_eq_state(self) -> bool:
        return True

    def eq_state(self) -> State:
        return np.array([-0.01])

    def _phase2d_setups(self) -> list[Task.Phase2DSetup]:
        return [Task.Phase2DSetup("phase", self.plot_phase, Task.mk_get2d([self.P, self.P]))]

    def plot_phase(self, ax: plt.Axes):
        """(Position, Velocity) plot."""
        raise NotImplementedError("")

    def nom_pol_zero(self, x: State) -> Control:
        return np.zeros(self.nu)
