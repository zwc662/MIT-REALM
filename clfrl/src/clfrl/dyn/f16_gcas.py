import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from jax_f16.controllers.pid import F16PIDController
from jax_f16.f16 import F16
from jax_f16.f16_types import S
from jaxtyping import Float

from clfrl.dyn.angle_encoder import AngleEncoder
from clfrl.dyn.dyn_types import BState, Control, HFloat, PolObs, State, VObs
from clfrl.dyn.sim_cts_pbar import SimCtsPbar
from clfrl.dyn.task import Task
from clfrl.plotting.phase2d_utils import plot_x_bounds, plot_y_bounds
from clfrl.plotting.plot_utils import plot_boundaries
from clfrl.plotting.plotstyle import PlotStyle
from clfrl.utils.angle_utils import rotx, roty, rotz
from clfrl.utils.costconstr_utils import poly4_clip_max_flat, poly4_softclip_flat
from clfrl.utils.jax_types import Arr, BBFloat, Vec3
from clfrl.utils.jax_utils import in_bounds, jax_jit, normalize_minmax, rep_vmap
from clfrl.utils.shape_utils import assert_shape


class F16GCAS(Task):
    NX = F16.NX
    # Assume that the throttle is set to
    NU = F16.NU - 1

    VT, ALPHA, BETA, PHI, THETA, PSI, P, Q, R, PN, PE, H, POW, NZINT, PSINT, NYRINT = range(NX)
    NZ, PS, NYR = range(NU)

    DT = 0.04

    def __init__(self):
        #                 [    Nz   Ps    Ny+R  ]
        self._u_min = np.array([-1.0, -5.0, -5.0])
        self._u_max = np.array([6.0, 5.0, 5.0])

        self._alt_max = 700

        self._dt = F16GCAS.DT

        self.f16 = F16()
        self.thrtl_nom = F16.trim_control()[F16.THRTL]

    @property
    def u_min(self):
        return self._u_min

    @property
    def u_max(self):
        return self._u_max

    @property
    def n_Vobs(self) -> int:
        return 20

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def x_labels(self) -> list[str]:
        return f16_xlabels()

    @property
    def u_labels(self) -> list[str]:
        return f16_ulabels()[:3]

    @property
    def h_labels(self) -> list[str]:
        return [
            "floor",
            "ceil",
            ##############3
            r"$\alpha_l$",
            r"$\alpha_u$",
            r"$\beta_l$",
            r"$\beta_u$",
            ##############3
            r"$nyr_l$",
            r"$nyr_u$",
            r"$\theta_l$",
            r"$\theta_u$",
        ]

    @property
    def h_max(self) -> float:
        return 1.0

    @property
    def h_min(self) -> float:
        return -1.0

    @property
    def max_ttc(self) -> float:
        return 6.0

    def h_components(self, state: State) -> HFloat:
        alt = state[F16.H]
        h_floor = -alt
        h_ceil = -(self._alt_max - alt)

        a_buf, b_buf = 5e-2, 5e-2
        h_alpha_lb = -(state[F16.ALPHA] - (A_BOUNDS[0] + a_buf))
        h_alpha_ub = -(A_BOUNDS[1] - a_buf - state[F16.ALPHA])

        h_beta_lb = -(state[F16.BETA] - (B_BOUNDS[0] + b_buf))
        h_beta_ub = -(B_BOUNDS[1] - b_buf - state[F16.BETA])

        h_nyr_lb = -(state[F16.NYRINT] - (-2.5))
        h_nyr_ub = -(2.5 - state[F16.NYRINT])

        # Avoid the euler angle singularity.
        h_theta_l = -(state[F16.THETA] - (-1.2))
        h_theta_u = -(1.2 - state[F16.THETA])

        coef_alpha, coef_beta = 1 / 0.08, 1 / 0.07
        coef_nyr, coef_theta = 1 / 0.2, 1 / 0.2

        hs_alt = jnp.array([h_floor / 100.0, h_ceil / 100.0])
        hs_alphabeta = jnp.array(
            [
                coef_alpha * h_alpha_lb,
                coef_alpha * h_alpha_ub,
                coef_beta * h_beta_lb,
                coef_beta * h_beta_ub,
            ]
        )
        hs_statebounds = jnp.array(
            [coef_nyr * h_nyr_lb, coef_nyr * h_nyr_ub, coef_theta * h_theta_l, coef_theta * h_theta_u]
        )
        # h <= 2
        hs_alt = poly4_clip_max_flat(hs_alt, max_val=2.0)
        # h <= 2
        hs_alphabeta = poly4_clip_max_flat(hs_alphabeta, max_val=1.0)
        # h <= 1
        hs_statebounds = poly4_clip_max_flat(hs_statebounds)
        hs = jnp.concatenate([hs_alt, hs_alphabeta, hs_statebounds])

        # softclip the minimum.
        hs = -poly4_softclip_flat(-hs, m=0.3)
        # hardclip the minimum.
        hs = -poly4_clip_max_flat(-hs, max_val=-self.h_min)

        return hs

    def _get_obs(self, state: State) -> VObs:
        """
        F16 has
            11 non-angles (VT, P, Q, R, pn, pe, h, pow, nzint, psint, ny+rint)
            5 angles (alpha, beta, phi, theta, psi)

        Observation:
             0 - 8:    [VT P Q R H Pow NzInt PsInt Ny+rInt]         ( 9, )
            9 - 16:    F16 angle cos + sin.                         ( 8, )
           17 - 19:    relative velocity vector                     ( 3, )
        """
        x = state
        nonangle_idxs = [self.VT, self.P, self.Q, self.R, self.H, self.POW, self.NZINT, self.PSINT, self.NYRINT]
        non_angle_obs = jnp.array([x[idx] for idx in nonangle_idxs])
        angles = np.array([self.ALPHA, self.BETA, self.PHI, self.THETA])
        angle_encoder = AngleEncoder(F16.NX, angles)
        angle_obs = angle_encoder.encode_angles(state)
        vel_vector = compute_f16_vel_angles(state)
        obs = jnp.concatenate([non_angle_obs, angle_obs, vel_vector])
        assert_shape(obs, self.n_Vobs, "full_observation")

        # fmt: off
        #                   [ VT        P           Q           R          H          POW       NZInt     PSInt     NY+RInt     alpha     alpha        beta       beta        phi        phi       theta     theta         psi       psi
        obs_min = np.array([2.13e+02, -4.05e-02, -5.07e-01, -5.14e-01,  4.79e+02,  2.12e+00, -1.87e+00, -3.44e-03, -2.20e-01,  8.57e-01, -9.25e-02,  9.60e-01, -1.80e-01,  9.99e-01, -3.75e-02,  7.41e-01, -1.26e-01,  9.01e-01, -4.26e-01, -2.20e-01])
        obs_max = np.array([5.42e+02,  3.90e-02,  5.78e-01,  4.95e-01,  6.18e+02,  8.42e+00,  1.14e+01,  1.43e-02,  6.89e-02,  1.00e+00,  5.15e-01,  1.00e+00,  2.04e-01,  1.00e+00,  3.04e-02,  1.00e+00,  6.71e-01,  1.00e+00,  4.22e-01,  9.22e-02])
        # fmt: on

        # Normalize observation.
        obs = normalize_minmax(obs, obs_min, obs_max)
        return obs

    def get_obs(self, state: State) -> tuple[VObs, PolObs]:
        obs = self._get_obs(state)
        return obs, obs

    def xdot(self, state: State, control: Control) -> State:
        self.chk_x(state)
        self.chk_u(control)

        kp_vt = 0.25
        throttle = -kp_vt * (state[self.VT] - 502.0)
        throttle = jnp.clip(throttle, 0, 1)
        # throttle = self.thrtl_nom
        control_with_thrtl = jnp.concatenate([control, jnp.array([throttle])])

        # Clip alpha and beta.
        a_buf, b_buf = 5e-3, 5e-3
        a_buf2, b_buf2 = 1e-1, 1e-1
        # a_buf, b_buf = 0.0, 0.0
        alpha_safe = jnp.clip(state[self.ALPHA], A_BOUNDS[0] - a_buf, A_BOUNDS[1] + a_buf)
        beta_safe = jnp.clip(state[self.BETA], B_BOUNDS[0] - b_buf, B_BOUNDS[1] + b_buf)
        state_safe = state.at[self.ALPHA].set(alpha_safe).at[self.BETA].set(beta_safe)

        xdot = self.f16.xdot(state_safe, control_with_thrtl)

        # However, if alpha is just too large, then just set xdot to 0.
        alpha_in_bounds = in_bounds(state[self.ALPHA], A_BOUNDS[0] - a_buf2, A_BOUNDS[1] + a_buf2)
        beta_in_bounds = in_bounds(state[self.BETA], B_BOUNDS[0] - b_buf2, B_BOUNDS[1] + b_buf2)
        is_in_bounds = alpha_in_bounds & beta_in_bounds
        xdot = jnp.where(is_in_bounds, xdot, 0.0)

        # # TODO: To prevent the controller from going crazy, if we are outside the morelli bounds, then set xdot=0.
        # alpha_safe = in_bounds(state[self.ALPHA], A_BOUNDS[0] - a_buf, A_BOUNDS[1] + a_buf)
        # beta_safe = in_bounds(state[self.BETA], B_BOUNDS[0] - b_buf, B_BOUNDS[1] + b_buf)
        # is_safe = alpha_safe & beta_safe
        # xdot = self.f16.xdot(state, control_with_thrtl)
        # xdot_safe = jnp.where(is_safe, xdot, 0.0)
        return xdot

    def f(self, state: State) -> State:
        self.chk_x(state)
        u_nom = np.array([0.0, 0.0, 0.0, self.thrtl_nom])
        return self.f16.xdot(state, u_nom)

    def G(self, state: State) -> State:
        self.chk_x(state)
        G = np.zeros((self.NX, self.NU))
        # Remember to map from [-1, 1] to control lims.
        G[self.NZINT, self.NZ] = -1.0
        G[self.PSINT, self.PS] = -1.0
        G[self.NYRINT, self.NYR] = -1.0
        return G

    def has_eq_state(self) -> bool:
        return True

    def eq_state(self) -> State:
        return self.f16.trim_state()

    def nominal_val_state(self) -> State:
        return self.f16.trim_state()

    def train_bounds(self) -> Float[Arr, "2 nx"]:
        lb = F16.state(
            250.0,
            [A_BOUNDS[0], B_BOUNDS[0]],
            [-np.pi, 0.9 * -np.pi / 2, -1e-4],
            [-4.0, -1.0, -np.pi],
            [0.0, 0.0, -100.0],
            0.0,
            [-5.0, -3.0, -3.0],
        )
        ub = F16.state(
            550.0,
            [A_BOUNDS[1], B_BOUNDS[1]],
            [np.pi, 0.9 * np.pi / 2, 1e-4],
            [4.0, 1.0, np.pi],
            [0.0, 0.0, 800.0],
            100.0,
            [15.0, 3.0, 3.0],
        )
        return np.stack([lb, ub], axis=0)

    def contour_bounds(self) -> Float[Arr, "2 nx"]:
        return self.train_bounds()

    def get_metric_x0(self) -> BState:
        return super().get_metric_x0()

    def get_ci_x0(self, setup: int = 0, n_pts: int = 80):
        return self.get_contour_x0(setup, n_pts)

    def get_loss_x0(self, n_sample: int = 128) -> BState:
        b_x0 = self.sample_train_x0(jr.PRNGKey(314159), n_sample)
        return b_x0

    def _phase2d_setups(self) -> list[Task.Phase2DSetup]:
        return [
            Task.Phase2DSetup("altpitch", self.plot_altpitch, Task.mk_get2d([self.H, self.THETA])),
            Task.Phase2DSetup("alphabeta", self.plot_alphabeta, Task.mk_get2d([self.BETA, self.ALPHA])),
            Task.Phase2DSetup("alpha_vt", self.plot_alphavt, Task.mk_get2d([self.ALPHA, self.VT])),
        ]

    def plot_altpitch(self, ax: plt.Axes):
        # x is altitude, y is pitch=theta
        train_bounds = self.train_bounds()
        PLOT_X_MIN, PLOT_X_MAX = train_bounds[:, self.H]
        PLOT_Y_MIN, PLOT_Y_MAX = train_bounds[:, self.THETA]
        ax.set(xlim=(PLOT_X_MIN, PLOT_X_MAX), ylim=(PLOT_Y_MIN, PLOT_Y_MAX))

        # Plot the avoid set.
        plot_x_bounds(ax, (0.0, self._alt_max), PlotStyle.obs_region)
        plot_y_bounds(ax, (-1.2, 1.2), PlotStyle.obs_region)
        ax.set(xlabel=r"alt (ft)", ylabel=r"$\theta$ (rad)")

    def plot_alphabeta(self, ax: plt.Axes):
        train_bounds = self.train_bounds()
        PLOT_X_MIN, PLOT_X_MAX = train_bounds[:, self.BETA]
        PLOT_Y_MIN, PLOT_Y_MAX = train_bounds[:, self.ALPHA]
        ax.set(xlim=(PLOT_X_MIN, PLOT_X_MAX), ylim=(PLOT_Y_MIN, PLOT_Y_MAX))

        # Plot the avoid set.
        plot_x_bounds(ax, B_BOUNDS, PlotStyle.obs_region)
        plot_y_bounds(ax, A_BOUNDS, PlotStyle.obs_region)
        ax.set(xlabel=r"$\beta$ (rad)", ylabel=r"$\alpha$ (rad)")

    def plot_alphavt(self, ax: plt.Axes):
        # x = alpha, y = vt
        train_bounds = self.train_bounds()
        PLOT_X_MIN, PLOT_X_MAX = train_bounds[:, self.ALPHA]
        PLOT_Y_MIN, PLOT_Y_MAX = train_bounds[:, self.VT]
        ax.set(xlim=(PLOT_X_MIN, PLOT_X_MAX), ylim=(PLOT_Y_MIN, PLOT_Y_MAX))

        # Plot the avoid set.
        plot_x_bounds(ax, A_BOUNDS, PlotStyle.obs_region)
        ax.set(xlabel=r"$\alpha$ (rad)", ylabel=r"$v_T$")

    def plot_boundaries(self, axes: list[plt.Axes]):
        axes_bounded = [axes[self.ALPHA], axes[self.BETA], axes[self.H], axes[self.THETA], axes[self.NYRINT]]
        lb = np.array([A_BOUNDS[0], B_BOUNDS[0], 0.0, -1.2, -2.5])
        ub = np.array([A_BOUNDS[1], B_BOUNDS[1], self._alt_max, 1.2, 2.5])
        bounds = np.stack([lb, ub], axis=0)
        plot_boundaries(axes_bounded, bounds, color="C0")

    def nom_pol_pid(self, state: State):
        nom_alt = self.nominal_val_state()[F16.H]
        pid_controller = F16PIDController(nom_alt)
        full_control = pid_controller.get_control(state)
        control = full_control[: self.NU]
        return jnp.clip(control, self.u_min, self.u_max)

    def get_bb_V_noms(self) -> dict[str, BBFloat]:
        tf = 6.0
        n_steps = int(round(tf / self.dt))
        sim = SimCtsPbar(self, self.nom_pol_pid, n_steps, self.dt, use_pid=True, max_steps=n_steps, solver="bosh3")

        out = {}
        for ii, setup in enumerate(self.phase2d_setups()):
            bb_x0, bb_Xs, bb_Ys = self.get_contour_x0(ii)
            bbT_x, _ = jax_jit(rep_vmap(sim.rollout_plot, rep=2))(bb_x0)
            bbT_h = jax_jit(rep_vmap(self.h, rep=3))(bbT_x)
            out[setup.plot_name] = bbT_h.max(axis=2)

        return out


A_BOUNDS = np.array([-0.17453292519943295, 0.7853981633974483])
B_BOUNDS = np.array([-0.5235987755982988, 0.5235987755982988])


def compute_f16_vel_angles(state: State) -> Vec3:
    """Compute cos / sin of [gamma, sigma], the pitch & yaw of the velocity vector."""
    assert state.shape == (F16.NX,)
    # 1: Compute {}^{W}R^{F16}.
    R_W_F16 = rotz(state[S.PSI]) @ roty(state[S.THETA]) @ rotx(state[S.PHI])
    assert R_W_F16.shape == (3, 3)

    # 2: Compute v_{F16}
    ca, sa = jnp.cos(state[S.ALPHA]), jnp.sin(state[S.ALPHA])
    cb, sb = jnp.cos(state[S.BETA]), jnp.sin(state[S.BETA])
    v_F16 = jnp.array([ca * cb, sb, sa * cb])

    # 3: Compute v_{W}
    v_W = R_W_F16 @ v_F16
    assert v_W.shape == (3,)

    # 4: Back out cos and sin of gamma and sigma.
    cos_sigma = v_W[0]
    sin_sigma = v_W[1]
    sin_gamma = v_W[2]

    out = jnp.array([cos_sigma, sin_sigma, sin_gamma])
    assert out.shape == (3,)
    return out


def f16_xlabels():
    return [
        r"$V_t$",
        r"$\alpha$",
        r"$\beta$",
        r"$\phi$",
        r"$\theta$",
        r"$\psi$",
        r"$P$",
        r"$Q$",
        r"$R$",
        r"$Pn$",
        r"$Pe$",
        r"alt",
        r"pow",
        r"$Nz$",
        r"$Ps$",
        r"$Ny+R$",
    ]


def f16_ulabels():
    return [r"$Nz$", r"$Ps$", r"$Ny+R$", r"thl"]
