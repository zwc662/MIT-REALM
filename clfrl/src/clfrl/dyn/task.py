import functools as ft
from abc import ABC, abstractmethod
from typing import Callable, NamedTuple

import einops as ei
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Float
from matplotlib import pyplot as plt

from clfrl.dyn.dyn_types import BState, Control, Disturb, HFloat, LFloat, PolObs, State, TState, VObs
from clfrl.dyn.odeint import rk4, tsit5
from clfrl.utils.jax_types import Arr, BFloat, BoolScalar, FloatScalar, TFloat
from clfrl.utils.jax_utils import jax_vmap
from clfrl.utils.mathtext import from_mathtext
from clfrl.utils.none import get_or
from clfrl.utils.rng import PRNGKey
from clfrl.utils.sampling_utils import get_mesh_np
from clfrl.utils.shape_utils import assert_shape


class Task(ABC):
    NX = None
    NU = None
    ND = None

    class mk_get2d:
        def __init__(self, idxs: list[int] | tuple[int, int]):
            assert len(idxs) == 2, "[mk_get2d] len(idxs) != 2 {}".format(len(idxs))
            self.idxs = tuple(idxs)

        def __call__(self, states: Float[Arr, "* nx"]):
            if states.shape[-1] == 2:
                return states

            x_idx, y_idx = self.idxs
            x, y = states[..., x_idx], states[..., y_idx]

            if isinstance(states, np.ndarray):
                return np.stack([x, y], axis=-1)
            else:
                return jnp.stack([x, y], axis=-1)

    class Phase2DSetup(NamedTuple):
        plot_name: str
        plot_avoid_fn: Callable
        # Get the 2d coordinates projections from a state.
        get2d_fn: "Task.mk_get2d" | Callable[[State], tuple[FloatScalar, FloatScalar]]
        setup_idx: int | None = None

        @property
        def idxs(self):
            return self.get2d_fn.idxs

        @property
        def idx0(self) -> int:
            return self.idxs[0]

        @property
        def idx1(self) -> int:
            return self.idxs[1]

        def plot(self, ax: plt.Axes, *args, **kwargs):
            return self.plot_avoid_fn(ax, *args, **kwargs)

    @property
    def name(self):
        return self.__class__.__name__

    def chk_x(self, state: State):
        return assert_shape(state, self.nx, "state")

    def chk_u(self, control: Control):
        return assert_shape(control, self.nu, "control")

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
    def u_min(self):
        return np.full(self.nu, -1.0)

    @property
    def u_max(self):
        return np.full(self.nu, +1.0)

    @property
    def n_polobs(self) -> int:
        return self.n_Vobs

    @property
    def dt(self) -> int:
        raise NotImplementedError("should have dt")

    @property
    @abstractmethod
    def n_Vobs(self) -> int:
        ...

    def goal_pt(self) -> State:
        return np.array([np.pi, 0.0])

    def l_components(self, state: State) -> LFloat:
        ...

    @property
    @abstractmethod
    def h_max(self) -> float:
        ...

    def h_components(self, state: State) -> HFloat:
        ...

    def h(self, state: State) -> FloatScalar:
        return self.h_components(state).max()

    def is_stable(self, T_state: TState) -> BoolScalar:
        ...

    def get_obs(self, state: State) -> tuple[VObs, PolObs]:
        ...

    def f(self, state: State) -> State:
        ...

    def G(self, state: State):
        ...

    def xdot(self, state: State, control: Control) -> State:
        self.chk_x(state)
        self.chk_u(control)
        control = control.clip(-1, 1)
        f, G = self.f(state), self.G(state)
        self.chk_x(f)
        Gu = G @ control
        self.chk_x(Gu)
        dx = f + Gu
        return self.chk_x(dx)

    def step(self, state: State, control: Control, disturb: Disturb = None) -> State:
        xdot_with_u = ft.partial(self.xdot, control=control)
        return rk4(self.dt, xdot_with_u, state)

    def step_plot(
        self, state: State, control: Control, disturb: Disturb = None, dt: float = None
    ) -> tuple[TState, TFloat]:
        xdot_with_u = ft.partial(self.xdot, control=control)
        dt = get_or(dt, self.dt)
        return tsit5(dt, 4, xdot_with_u, state), np.linspace(0, dt, num=5)

    def has_eq_state(self) -> bool:
        return False

    def eq_state(self) -> State:
        ...

    def assert_is_safe(self, state: State) -> BoolScalar:
        raise NotImplementedError("")

    def train_bounds(self) -> Float[Arr, "2 nx"]:
        ...

    def contour_bounds(self) -> Float[Arr, "2 nx"]:
        return self.train_bounds()

    def sample_train_x0(self, key: PRNGKey, n_sample: int) -> BState:
        x_lb, x_ub = self.train_bounds()
        return jr.uniform(key, (n_sample, self.nx), minval=x_lb, maxval=x_ub)

    def get_plot_x0(self, setup_idx: int = 0) -> BState:
        with jax.ensure_compile_time_eval():
            n_pts = 13
            setup = self.phase2d_setups()[setup_idx]
            bb_Xs, bb_Ys, bb_x0 = get_mesh_np(self.train_bounds(), setup.idxs, n_pts, n_pts, self.nominal_val_state())
            b_x0 = ei.rearrange(bb_x0, "nys nxs nx -> (nys nxs) nx")

            # Only keep the ones that are inside.
            b_in_ci = jax_vmap(self.in_ci_approx)(b_x0)
            b_x0 = b_x0[b_in_ci]

        return b_x0

    def get_metric_x0(self) -> BState:
        with jax.ensure_compile_time_eval():
            n_pts, idxs = 32, (0, 1)
            bb_Xs, bb_Ys, bb_x0 = get_mesh_np(self.train_bounds(), idxs, n_pts, n_pts, self.nominal_val_state())
            b_x0 = ei.rearrange(bb_x0, "nys nxs nx -> (nys nxs) nx")

            # Only keep the ones that are inside.
            b_in_ci = jax_vmap(self.in_ci_approx)(b_x0)
            b_x0 = b_x0[b_in_ci]

        return b_x0

    def get_loss_x0(self, n_sample: int = 512) -> BState:
        b_x0 = self.sample_train_x0(jr.PRNGKey(314159), n_sample)
        return b_x0

    def get_plot_rng_x0(self) -> State:
        ...

    def get_contour_x0(self, setup: int = 0, n_pts: int = 80):
        with jax.ensure_compile_time_eval():
            phase2d_setup = self.phase2d_setups()[setup]
            idxs = phase2d_setup.get2d_fn.idxs
            bb_Xs, bb_Ys, bb_x0 = get_mesh_np(self.contour_bounds(), idxs, n_pts, n_pts, self.nominal_val_state())
        return bb_x0, bb_Xs, bb_Ys

    def in_ci_approx(self, state: State) -> BoolScalar:
        EPS = 1e-4
        h_comps = self.h_components(state)
        return h_comps.max() < EPS

    def nominal_val_state(self) -> State:
        ...

    @property
    @abstractmethod
    def x_labels(self) -> list[str]:
        ...

    @property
    @abstractmethod
    def u_labels(self) -> list[str]:
        ...

    @property
    def l_labels(self) -> list[str]:
        return []

    @property
    @abstractmethod
    def h_labels(self) -> list[str]:
        ...

    @property
    def h_labels_clean(self):
        return [from_mathtext(label) for label in self.h_labels]

    def phase2d_setups(self) -> list[Phase2DSetup]:
        setups = self._phase2d_setups()
        setup: Task.Phase2DSetup
        for ii, setup in enumerate(setups):
            setups[ii] = setup._replace(setup_idx=ii)
        return setups

    def _phase2d_setups(self) -> list[Phase2DSetup]:
        ...

    def plot(self, ax: plt.Axes, setup_idx: int):
        name, plot_fn, get2d, _ = self.phase2d_setups()[setup_idx]
        plot_fn(ax)
