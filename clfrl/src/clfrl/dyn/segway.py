import functools as ft

import control as ct
import einops as ei
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy
from attrs import define
from jaxtyping import Float
from loguru import logger

from clfrl.dyn.dyn_types import BState, Control, Disturb, HFloat, LFloat, PolObs, State, TState, VObs
from clfrl.dyn.odeint import rk4, tsit5
from clfrl.dyn.task import Task
from clfrl.networks.fourier_emb import pos_embed_random
from clfrl.plotting.phase2d_utils import plot_x_bounds
from clfrl.plotting.plotstyle import PlotStyle
from clfrl.utils.costconstr_utils import add_constr_margin, poly4_clip_max_flat
from clfrl.utils.jax_types import Arr, BoolScalar, TFloat
from clfrl.utils.jax_utils import jax2np, jax_vmap
from clfrl.utils.none import get_or
from clfrl.utils.sampling_utils import get_mesh_np
from clfrl.utils.small_la import inv22


class Segway(Task):
    NX = 4
    NU = 1

    P, TH, V, W = range(NX)
    (F,) = range(NU)

    @define
    class Params:
        m: float = 0.5
        M: float = 1.5
        J: float = 0.01
        l: float = 1.0
        g: float = 9.81
        # Linear friction
        c: float = 0.1
        # Angular friction
        gamma: float = 0.1

    def __init__(self, p=Params()):
        self.p = p
        self.umax = 40.0
        self._theta_max = 0.3 * np.pi
        self.dt_mult = 10.0

        self._dt_orig = 0.01
        self._p_max = 2.0

        # self._dt_orig = 0.004
        # self._p_max = 5.0

        self._dt = self._dt_orig * self.dt_mult

        x_eq, u_eq = self.eq_pt()
        Q = np.diag([1.0, 1.0, 1.0, 1.0])
        R = 0.5 * np.diag(np.ones(self.nu))

        logger.info("Getting jac of f")
        A, B = jax2np(jax.jacobian(self.f)(x_eq)), jax2np(self.G(x_eq))
        logger.info("Solving continuous are...")
        S = scipy.linalg.solve_continuous_are(A, B, Q, R)
        logger.info("Solving continuous are... Done!")
        self.lqr_K = np.linalg.inv(R) @ B.T @ S
        self.n_posemb_feats = 20

    @property
    def n_Vobs(self) -> int:
        # # [p, sin, cos, v, w]
        # return 5
        # # [sin, cos, v, w]
        # return 4
        # [sin; cos; pos_emb(p); pos_emb(v); w]
        return 2 + self.n_posemb_feats + self.n_posemb_feats + 1

    @property
    def x_labels(self) -> list[str]:
        return [r"$p$", r"$\theta$", r"$v$", r"$\omega$"]

    @property
    def u_labels(self) -> list[str]:
        return [r"$F$"]

    @property
    def h_labels(self) -> list[str]:
        return [r"$\theta_l$", r"$\theta_u$", r"$p_l$", r"$p_u$"]

    @property
    def h_max(self) -> float:
        # return 0.75
        return 1.0

    @property
    def h_unsafe_lb(self) -> float:
        return 0.5

    @property
    def h_safe_ub(self) -> float:
        return -0.5

    def h_components(self, state: State) -> HFloat:
        p, th, v, w = self.chk_x(state)
        theta_lb = th - self._theta_max
        theta_ub = -(th + self._theta_max)
        p_lb = p - self._p_max
        p_ub = -(p + self._p_max)

        # h <= 1
        hs = poly4_clip_max_flat(jnp.array([theta_lb, theta_ub, p_lb, p_ub]))
        return hs
        # # Make sure h is bounded.
        # # [..., 1] -> [..., 0.25]
        # hs = jnp.clip(0.25 * hs, a_max=self.h_max - 0.5)
        # # [..., 0.25] -> [..., 0.75]
        # # Add a margin. The ub (lb) of the safe (unsafe) are h_safe_ub (h_unsafe_lb).
        # return add_constr_margin(hs, 1.0)

    def get_obs(self, state: State) -> tuple[VObs, PolObs]:
        p, th, v, w = self.chk_x(state)
        sin, cos = jnp.sin(th), jnp.cos(th)

        # Position embedding for p and v.
        p_emb = pos_embed_random(self.n_posemb_feats // 2, p, scale=0.5, seed=897651)
        v_emb = pos_embed_random(self.n_posemb_feats // 2, v / 5.0, scale=0.5, seed=897651)

        sincos = jnp.array([sin, cos])
        w_obs = jnp.array([w]) / 5.0
        obs = jnp.concatenate([sincos, p_emb, v_emb, w_obs], axis=0)
        assert obs.shape == (self.n_Vobs,)
        return obs, obs

    @property
    def dt(self):
        # return self._dt
        return self._dt_orig * self.dt_mult

    def M_mat(self, state: State):
        p, th, v, w = self.chk_x(state)
        M_11 = self.p.m + self.p.M
        M_12 = M_21 = -self.p.m * self.p.l * jnp.cos(th)
        M_22 = self.p.J + self.p.m * self.p.l**2
        return jnp.array([[M_11, M_12], [M_21, M_22]])

    def f(self, state: State) -> State:
        p = self.p
        _, th, v, w = self.chk_x(state)
        sin, cos = jnp.sin(th), jnp.cos(th)

        Ctau = jnp.array([p.c * v + p.m * p.l * sin * w**2, p.gamma * w - p.m * p.g * p.l * sin])
        M_inv = inv22(self.M_mat(state))
        F_vel = -M_inv @ Ctau
        # F_vel = jnp.linalg.solve(self.M_mat(state), -Ctau)
        assert F_vel.shape == (2,)
        F = jnp.concatenate([jnp.array([v, w]), F_vel], axis=0)
        assert F.shape == (self.nx,)
        return F / self.dt_mult

    def G(self, state: State):
        self.chk_x(state)
        B = np.array([[1.0, 0.0]]).T
        M_inv = inv22(self.M_mat(state))
        # G_vel = jnp.linalg.solve(self.M_mat(state), B)
        G_vel = M_inv @ B
        assert G_vel.shape == (2, self.nu)
        G = jnp.concatenate([jnp.zeros((2, self.nu)), G_vel], axis=0)
        assert G.shape == (self.nx, self.nu)
        return G * self.umax / self.dt_mult

    def step(self, state: State, control: Control, disturb: Disturb = None) -> State:
        xdot_with_u = ft.partial(self.xdot, control=control)
        return rk4(self.dt, xdot_with_u, state)

    def step_plot(
        self, state: State, control: Control, disturb: Disturb = None, dt: float = None
    ) -> tuple[TState, TFloat]:
        xdot_with_u = ft.partial(self.xdot, control=control)
        dt = get_or(dt, self.dt)
        return tsit5(dt, 6, xdot_with_u, state), np.linspace(0, dt, num=7)

    def train_bounds(self) -> Float[Arr, "2 nx"]:
        # [p th v w]
        return np.array([(-2.5, 2.5), (-1.8, 1.8), (-6.5, 6.5), (-5.8, 5.8)]).T

    def contour_bounds(self) -> Float[Arr, "2 nx"]:
        # [p th v w]
        return np.array([(-2.5, 2.5), (-1.8, 1.8), (-6.5, 6.5), (-5.8, 5.8)]).T

    def eq_pt(self):
        return np.zeros(self.nx), np.zeros(self.nu)

    def nominal_val_state(self):
        return np.zeros(self.nx)

    def _phase2d_setups(self) -> list[Task.Phase2DSetup]:
        return [
            Task.Phase2DSetup("pend", self.plot_pend, Task.mk_get2d([self.TH, self.W])),
            Task.Phase2DSetup("pv", self.plot_pv, Task.mk_get2d([self.P, self.V])),
        ]

    def plot_pend(self, ax: plt.Axes):
        """(Theta, Omega) plot."""
        PLOT_XMIN, PLOT_XMAX = -1.8, 1.8
        PLOT_YMIN, PLOT_YMAX = -5.8, 5.8
        ax.set(xlim=(PLOT_XMIN, PLOT_XMAX), ylim=(PLOT_YMIN, PLOT_YMAX))
        ax.set(xlabel=r"$\theta$", ylabel=r"$\omega$")

        # 2: Plot the avoid set.
        plot_x_bounds(ax, (-self._theta_max, self._theta_max), PlotStyle.obs_region)

    def plot_pv(self, ax: plt.Axes):
        """(p, v) plot."""
        PLOT_XMIN, PLOT_XMAX = -2.5, 2.5
        PLOT_YMIN, PLOT_YMAX = -6.5, 6.5
        ax.set(xlim=(PLOT_XMIN, PLOT_XMAX), ylim=(PLOT_YMIN, PLOT_YMAX))
        ax.set(xlabel=r"$p$", ylabel=r"$v$")

        # 2: Plot the avoid set.
        plot_x_bounds(ax, (-self._p_max, self._p_max), PlotStyle.obs_region)

    def nom_pol_lqr(self, state: State):
        self.chk_x(state)
        u = -self.lqr_K @ state
        return u.clip(-1, 1)
