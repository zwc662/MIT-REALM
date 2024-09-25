import functools as ft

import ipdb
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from jax_f16.controllers.pid import F16N0PIDController, F16PIDController
from jax_f16.f16 import F16
from jax_f16.f16_types import S
from jaxtyping import Float

from pncbf.dyn.angle_encoder import AngleEncoder
from pncbf.dyn.dyn_types import BState, Control, HFloat, PolObs, State, VObs
from pncbf.dyn.f16_gcas import compute_f16_vel_angles, f16_ulabels, f16_xlabels
from pncbf.dyn.task import Task
from pncbf.plotting.phase2d_utils import plot_x_bounds, plot_y_bounds
from pncbf.plotting.plot_utils import plot_boundaries, plot_boundaries_with_clip
from pncbf.plotting.plotstyle import PlotStyle
from pncbf.utils.angle_utils import rotx, roty, rotz, wrap_to_pi
from pncbf.utils.constraint_utils import to_vec_mag_exp
from pncbf.utils.costconstr_utils import poly4_clip_max_flat, poly4_softclip_flat
from pncbf.utils.jax_types import Arr, FloatScalar, RotMat3D, Vec3
from pncbf.utils.jax_utils import jax_vmap, normalize_minmax
from pncbf.utils.rng import PRNGKey
from pncbf.utils.sampling_utils import get_mesh_np
from pncbf.utils.sdf_utils import sdf_capped_cone


class F16Two(Task):
    """Two F16s. Other one is PID with all zeros.

    z is pointed down, so....
        North is 0
        East is pi/2
        South is pi.
        West is -pi/2
    """

    NX = 2 * F16.NX
    # Throttle PID.
    NU = F16.NU - 1

    VT0, ALPHA0, BETA0, PHI0, THETA0, PSI0, P0, Q0, R0, PN0, PE0, H0, POW0, NZINT0, PSINT0, NYRINT0 = np.arange(F16.NX)
    (
        VT1,
        ALPHA1,
        BETA1,
        PHI1,
        THETA1,
        PSI1,
        P1,
        Q1,
        R1,
        PN1,
        PE1,
        H1,
        POW1,
        NZINT1,
        PSINT1,
        NYRINT1,
    ) = F16.NX + np.arange(F16.NX)
    NZ, PS, NYR = range(NU)

    POS0 = F16.POS
    POS0_NEU = F16.POS_NEU
    POS1 = F16.POS + F16.NX
    POS1_NEU = F16.POS_NEU + F16.NX

    DT = 0.04

    def __init__(self):
        #                 [    Nz   Ps    Ny+R  ]
        self._u_min = np.array([-1.0, -5.0, -5.0])
        self._u_max = np.array([6.0, 5.0, 5.0])

        self._alt_max = 700
        self._dt = F16Two.DT

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
        return 27

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def x_labels(self) -> list[str]:
        x_labels = f16_xlabels()
        labels0 = [f"{label}0" for label in x_labels]
        labels1 = [f"{label}1" for label in x_labels]
        return labels0 + labels1

    @property
    def u_labels(self) -> list[str]:
        return f16_ulabels()[:3]

    @property
    def h_labels(self) -> list[str]:
        return [
            "floor",
            "ceil",
            ##############
            r"$\alpha_l$",
            r"$\alpha_u$",
            r"$\beta_l$",
            r"$\beta_u$",
            ##############
            r"$ps_l$",
            r"$ps_u$",
            r"$nyr_l$",
            r"$nyr_u$",
            ##############
            r"$\theta_l$",
            r"$\theta_u$",
            ##############
            "collide",
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

    def h_collide_fn(self, state: State) -> FloatScalar:
        self.chk_x(state)
        # 1: Compute p_1_0.
        R_W_1 = rotz(state[self.PSI1]) @ roty(state[self.THETA1]) @ rotx(state[self.PHI1])
        p_W_0m1 = state[self.POS0_NEU] - state[self.POS1_NEU]
        p_1_0 = R_W_1.T @ p_W_0m1

        # 2: Compute h.
        a = np.array([0.0, 0.0, 0.0])
        # b = np.array([0.0, 500.0, 0.0])
        b = np.array([-500.0, 0.0, 0.0])
        # Radius of cone at plane1
        ra = 40.0
        # Radius of cone at tail.
        rb = 80.0
        # Wake turbulence. Scale things down, since numbers will be large, we might be hitting precision issues.
        scale = 200
        sdf_wake = sdf_capped_cone(p_1_0 / scale, a / scale, b / scale, ra / scale, rb / scale) * scale
        # Model plane as a sphere.
        sdf_plane = jnp.linalg.norm(p_1_0 - np.array([-20.0, 0.0, 0.0]), axis=-1) - 42.0
        sdf = jnp.minimum(sdf_wake, sdf_plane)
        # Assume the ego F16 is a sphere of radius 30 ft.
        sdf = sdf - 30.0
        return -sdf

    def h_components(self, state: State) -> HFloat:
        self.chk_x(state)
        alt = state[F16.H]
        h_floor = -alt
        h_ceil = -(self._alt_max - alt)

        a_buf, b_buf = 5e-2, 5e-2
        h_alpha_lb = -(state[F16.ALPHA] - (A_BOUNDS[0] + a_buf))
        h_alpha_ub = -(A_BOUNDS[1] - a_buf - state[F16.ALPHA])

        h_beta_lb = -(state[F16.BETA] - (B_BOUNDS[0] + b_buf))
        h_beta_ub = -(B_BOUNDS[1] - b_buf - state[F16.BETA])

        h_ps_lb = -(state[F16.PSINT] - (-2.5))
        h_ps_ub = -(2.5 - state[F16.PSINT])

        h_nyr_lb = -(state[F16.NYRINT] - (-2.5))
        h_nyr_ub = -(2.5 - state[F16.NYRINT])

        # Avoid the euler angle singularity.
        h_theta_l = -(state[F16.THETA] - (-1.2))
        h_theta_u = -(1.2 - state[F16.THETA])

        # Avoid collision
        h_collide = self.h_collide_fn(state) / 120.0

        coef_alpha, coef_beta = 1 / 0.08, 1 / 0.07
        coef_ps, coef_nyr, coef_theta = 1 / 0.2, 1 / 0.2, 1 / 0.2

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
            [
                coef_ps * h_ps_lb,
                coef_ps * h_ps_ub,
                coef_nyr * h_nyr_lb,
                coef_nyr * h_nyr_ub,
                coef_theta * h_theta_l,
                coef_theta * h_theta_u,
            ]
        )
        # h <= 2
        hs_alt = poly4_clip_max_flat(hs_alt, max_val=2.0)
        # h <= 1
        hs_alphabeta = poly4_clip_max_flat(hs_alphabeta, max_val=1.0)
        # h <= 1
        hs_statebounds = poly4_clip_max_flat(hs_statebounds)
        h_collide = poly4_clip_max_flat(h_collide)
        hs = jnp.concatenate([hs_alt, hs_alphabeta, hs_statebounds], axis=0)

        # softclip the minimum.
        hs = -poly4_softclip_flat(-hs, m=0.3)

        h_collide = -poly4_softclip_flat(-h_collide, m=0.4)
        h_collide = -poly4_softclip_flat(-h_collide, m=0.3)

        hs = jnp.concatenate([hs, h_collide[None]], axis=0)

        # hardclip the minimum.
        hs = -poly4_clip_max_flat(-hs, max_val=-self.h_min)

        return hs

    @property
    def obs_labels(self) -> list[str]:
        labels = [
            "vT",
            "P",
            "Q",
            "R",
            "H",
            "Pow",
            "NzInt",
            "PsInt",
            "Ny+RInt",
            r"$\cos\alpha$",
            r"$\sin\alpha$",
            r"$\cos\beta$",
            r"$\sin\beta$",
            r"$\cos\phi$",
            r"$\sin\phi$",
            r"$\cos\theta$",
            r"$\sin\theta$",
            "vx",
            "vy",
            "vz",
            "dpx",
            "dpy",
            "dpz",
            r"$|dp|$",
            "dvx",
            "dvy",
            "dvz",
            # r"$|dv|$",
        ]
        assert len(labels) == self.n_Vobs
        return labels

    def _get_obs(self, state: State) -> VObs:
        """
        F16 has
            11 non-angles (VT, P, Q, R, pn, pe, h, pow, nzint, psint, ny+rint)
            5 angles (alpha, beta, phi, theta, psi)

        We also include observations of relative positions and velocities for the other F16.
            {}^W p^B_{v_A}:                         (3, )
                 v^{v_B}_{v_A} - v^{v_A}_{v_A} :    (3, )
        However, to encode the inductive bias that smaller distances matter more than larger distances, we encode the
        vector as a unit vector and the exponentially transformed magnitude.

        Observation:
             0 - 8:    [VT P Q R H Pow NzInt PsInt Ny+rInt]         ( 9, )
            9 - 16:    F16 angle cos + sin.                         ( 8, )
           17 - 19:    relative velocity vector                     ( 3, )
           20 - 23:    {}^W p^B_{v_A}                               ( 4, )
           24 - 27:    v^{v_B}_{v_A} - v^{v_A}_{v_A}                ( 4, )
        """
        x0 = state[: F16.NX]
        nonangle_idxs = [F16.VT, F16.P, F16.Q, F16.R, F16.H, F16.POW, F16.NZINT, F16.PSINT, F16.NYRINT]
        non_angle_obs = jnp.array([x0[idx] for idx in nonangle_idxs])
        angles = np.array([F16.ALPHA, F16.BETA, F16.PHI, F16.THETA])
        angle_encoder = AngleEncoder(F16.NX, angles)
        angle_obs = angle_encoder.encode_angles(x0)
        v_W_vA_unit = compute_f16_vel_angles(x0)
        self_obs = jnp.concatenate([non_angle_obs, angle_obs, v_W_vA_unit])

        # Compute relative position and velocity.
        x1 = state[F16.NX :]
        p_W_0m1 = get_pos_NED(x1) - get_pos_NED(x0)

        v_W_vB_unit = compute_f16_vel_angles(x1)
        v_W_vAmvB = v_W_vA_unit * x0[F16.VT] - v_W_vB_unit * x1[F16.VT]

        # Transform to x0's wind frame.
        R_W_Wnd0 = get_R_W_Wnd(x0)

        # The max relevant position observations should be around 2_000 ft?
        p_W_0m1_wnd0_obs = to_vec_mag_exp(R_W_Wnd0.T @ p_W_0m1, magnitude_scale=800.0)
        # The velocity vector magnitude at most be around 1_000.
        v_W_vAmvB_wnd0_obs = R_W_Wnd0.T @ v_W_vAmvB

        # (8, )
        other_obs = jnp.concatenate([p_W_0m1_wnd0_obs, v_W_vAmvB_wnd0_obs], axis=0)

        obs = jnp.concatenate([self_obs, other_obs], axis=0)
        assert obs.shape == (self.n_Vobs,)

        # fmt: off
        #                  [ VT        P           Q           R          H          POW       NZInt     PSInt     NY+RInt     alpha     alpha        beta       beta        phi        phi       theta     theta        vx        vy         vz           dpos_x     dpos_y     dpos_z     |dpos|, dv_x       dv_y       dv_z       ]
        obs_min = np.array(
            [1.93e+02, -3.57e+00, -1.97e+00, -2.09e+00, -4.80e+02, -9.82e+00, -1.69e+01, -3.29e+00, -2.81e+00,
             -6.71e-01, -8.52e-01, -8.88e-01, -9.18e-01, -9.96e-01, -9.99e-01, 1.84e-01, -9.38e-01, -9.86e-01,
             -9.85e-01, -8.28e-01, -9.96e-01, -8.99e-01, -9.30e-01, 3.10e-02, -2.02e+02, -4.52e+02,
             -4.80e+02])  # obs_min = np.array([1.95e+02, -3.52e+00, -1.93e+00, -2.04e+00, -4.10e+02, -9.79e+00, -1.68e+01, -3.21e+00, -2.78e+00, -6.37e-01, -8.45e-01, -8.87e-01, -9.19e-01, -9.96e-01, -9.99e-01,  1.73e-01, -9.41e-01, -9.36e-01, -9.99e-01, -8.17e-01, -2.27e+03, -1.02e+03, -1.61e+03, -2.01e+02, -4.43e+02, -4.69e+02])
        obs_max = np.array(
            [5.41e+02, 2.27e+00, 1.09e+00, 3.31e+00, 1.14e+03, 1.00e+02, 4.81e+00, 2.50e+00, 7.96e+00, 1.00e+00,
             9.85e-01, 1.00e+00, 9.73e-01, 9.98e-01, 9.99e-01, 1.00e+00, 8.95e-01, 9.85e-01, 9.93e-01, 9.48e-01,
             9.98e-01, 9.02e-01, 9.30e-01, 8.68e-01, 1.02e+03, 4.50e+02,
             4.81e+02])  # obs_max = np.array([5.41e+02, 2.29e+00, 1.08e+00, 3.28e+00, 1.05e+03, 1.00e+02, 4.81e+00, 2.47e+00, 7.65e+00, 1.00e+00, 9.84e-01, 1.00e+00, 9.68e-01, 9.98e-01, 9.99e-01, 1.00e+00, 8.84e-01, 9.06e-01, 9.99e-01, 9.46e-01, 2.02e+03, 1.12e+03, 1.05e+03, 1.03e+03, 4.25e+02, 4.74e+02])
        # # obs_min = np.array([+1.58e+02, -5.08e+00, -2.51e+00, -3.36e+00, -1.01e+03,  1.07e+01, -2.49e+01, -5.54e+00, -3.65e+00, -9.31e-01, -9.73e-01, -9.86e-01, -9.89e-01, -9.96e-01, -9.98e-01, -3.67e-01, -9.63e-01, -9.60e-01, -9.60e-01, -9.65e-01, -3.26e+03, -2.42e+03, -2.86e+03, -2.26e+02, -4.94e+02, -4.98e+02])
        # # obs_max = np.array([+5.08e+02, +2.92e+00, +1.45e+00, +3.52e+00, +1.77e+03, +1.00e+02, +8.98e+00, +4.26e+00, +9.29e+00, +1.00e+00, +9.93e-01, +1.00e+00, +9.84e-01, +9.99e-01, +9.97e-01, +1.00e+00, +9.64e-01, +9.64e-01, +9.57e-01, +9.68e-01, +2.36e+03, +2.80e+03, +2.36e+03, +8.72e+02, +4.98e+02, +4.96e+02])
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

        state0, state1 = state[: F16.NX], state[F16.NX :]

        #######################################################
        # Ego Dynamics.

        # Throttle PID.
        kp_vt = 0.25
        throttle = -kp_vt * (state0[F16.VT] - 502.0)
        throttle = jnp.clip(throttle, 0, 1)
        control_with_thrtl_0 = jnp.concatenate([control, jnp.array([throttle])])

        # Clip alpha and beta for ego.
        a_buf, b_buf = 5e-3, 5e-3
        a_buf2, b_buf2 = 1e-1, 1e-1
        alpha_safe = jnp.clip(state0[F16.ALPHA], A_BOUNDS[0] - a_buf, A_BOUNDS[1] + a_buf)
        beta_safe = jnp.clip(state0[F16.BETA], B_BOUNDS[0] - b_buf, B_BOUNDS[1] + b_buf)
        state_safe_0 = state0.at[F16.ALPHA].set(alpha_safe).at[F16.BETA].set(beta_safe)
        xdot_0 = self.f16.xdot(state_safe_0, control_with_thrtl_0)

        #######################################################
        # Other dynamics.
        # Throttle PID.
        kp_vt = 0.25
        throttle = -kp_vt * (state1[F16.VT] - 502.0)
        throttle = jnp.clip(throttle, 0, 1)
        control_with_thrtl_1 = jnp.concatenate([np.zeros(3), jnp.array([throttle])])

        # Clip alpha and beta for other as well.

        xdot_1 = self.f16.xdot(state1, control_with_thrtl_1)

        xdot = jnp.concatenate([xdot_0, xdot_1], axis=0)
        return self.chk_x(xdot)

    def f(self, state: State) -> State:
        self.chk_x(state)
        u_nom = np.zeros(3)
        return self.xdot(state, u_nom)

    def G(self, state: State) -> State:
        self.chk_x(state)
        G = np.zeros((self.NX, self.NU))
        G[self.NZINT0, self.NZ] = -1.0
        G[self.PSINT0, self.PS] = -1.0
        G[self.NYRINT0, self.NYR] = -1.0
        return G

    def has_eq_state(self) -> bool:
        return True

    def eq_state(self) -> State:
        # plane0 facing EAST, at (1200, 0).
        # plane1 facing WEST, at (0, 0).
        state0 = self.f16.trim_state()
        state1 = self.f16.trim_state()
        state0[F16.POS2D] = np.array([1200.0, 0.0])
        state0[F16.PSI] = np.pi / 2

        state1[F16.POS2D] = np.array([0.0, 0.0])
        state1[F16.PSI] = -np.pi / 2
        state = np.concatenate([state0, state1], axis=0)
        return state

    def nominal_val_state(self) -> State:
        return self.eq_state()

    def other_bounds(self):
        roll_frac = 0.3
        pitch_frac = 0.2
        trim = self.eq_state()[F16.NX :]
        lb1 = F16.state(
            450.0,
            [trim[F16.ALPHA] - 0.05, trim[F16.BETA] - 0.05],
            [-np.pi * roll_frac, -np.pi / 2 * pitch_frac, -np.pi - 1e-3],
            [-0.5, -0.5, -0.2 * np.pi],
            [-500.0, -100.0, 450.0],
            0.0,
            [-1.0, -1.0, -1.0],
        )
        ub1 = F16.state(
            550.0,
            [trim[F16.ALPHA] + 0.1, trim[F16.BETA] + 0.05],
            [np.pi * roll_frac, np.pi / 2 * pitch_frac, np.pi + 1e-3],
            [0.5, 0.5, 0.2 * np.pi],
            [500.0, 100.0, 550.0],
            15.0,
            [1.0, 1.0, 1.0],
        )
        bounds1 = np.stack([lb1, ub1], axis=0)
        return bounds1

    def train_bounds(self) -> Float[Arr, "2 nx"]:
        lb0 = F16.state(
            250.0,
            [A_BOUNDS[0], B_BOUNDS[0]],
            [-np.pi, 0.9 * -np.pi / 2, -np.pi],
            [-4.0, -1.0, -np.pi],
            [-400.0, -3_000.0, -100.0],
            0.0,
            [-5.0, -3.0, -3.0],
        )
        ub0 = F16.state(
            550.0,
            [A_BOUNDS[1], B_BOUNDS[1]],
            [np.pi, 0.9 * np.pi / 2, np.pi],
            [4.0, 1.0, np.pi],
            [400.0, 1_500.0, 800.0],
            100.0,
            [10.0, 3.0, 3.0],
        )

        # (2, f16_nx)
        bounds0_train = np.stack([lb0, ub0], axis=0)
        bounds_train = np.concatenate([bounds0_train, self.other_bounds()], axis=1)
        assert bounds_train.shape == (2, self.nx)
        return bounds_train

    def sample_train_x0_random(self, key: PRNGKey, n_sample: int) -> BState:
        x_lb, x_ub = self.train_bounds()
        b_x = jr.uniform(key, (n_sample, self.nx), minval=x_lb, maxval=x_ub)
        return b_x

    def sample_train_x0_infront(self, key: PRNGKey, n_sample: int) -> BState:
        key0, key_smooth, key_front_times = jr.split(key, 3)

        x_lb, x_ub = self.train_bounds()
        b_x = jr.uniform(key0, (n_sample, self.nx), minval=x_lb, maxval=x_ub)
        b_x0, b_x1 = b_x[:, : F16.NX], b_x[:, F16.NX :]

        # For this, smooth out the initial state of plane0.
        b_a = jr.uniform(key_smooth, (n_sample,), minval=0.0, maxval=0.5)
        x = self.eq_state()[: F16.NX]
        b_alpha_sm = (1 - b_a) * x[F16.ALPHA] + b_a * b_x[:, F16.ALPHA]
        b_beta_sm = (1 - b_a) * x[F16.BETA] + b_a * b_x[:, F16.BETA]
        b_theta_sm = (1 - b_a) * x[F16.THETA] + b_a * b_x[:, F16.THETA]
        b_p_sm = (1 - b_a) * x[F16.P] + b_a * b_x0[:, F16.P]
        b_q_sm = (1 - b_a) * x[F16.Q] + b_a * b_x0[:, F16.Q]
        b_r_sm = (1 - b_a) * x[F16.R] + b_a * b_x0[:, F16.R]
        b_h_sm = (1 - b_a) * b_x1[:, F16.H] + b_a * b_x0[:, F16.H]
        b_nzint_sm = (1 - b_a) * x[F16.NZINT] + b_a * b_x[:, F16.NZINT]
        b_x_sm = b_x.at[:, F16.ALPHA].set(b_alpha_sm).at[:, F16.BETA].set(b_beta_sm).at[:, F16.THETA].set(b_theta_sm)
        b_x_sm = b_x_sm.at[:, F16.P].set(b_p_sm).at[:, F16.Q].set(b_q_sm).at[:, F16.R].set(b_r_sm)
        b_x_sm = b_x_sm.at[:, F16.H].set(b_h_sm).at[:, F16.NZINT].set(b_nzint_sm)
        b_x = b_x_sm

        b_x0, b_x1 = b_x[:, : F16.NX], b_x[:, F16.NX :]

        # Sample positions that are directly in front of the other plane.
        b_velangle_1 = jax_vmap(compute_f16_vel_angles)(b_x1)
        b_times = jr.uniform(key_front_times, (n_sample,), minval=-1.0, maxval=6.0)
        # flip the z because NEU not NED.
        z_flip = np.array([1.0, 1.0, -1.0])
        b_x1_point = b_x1[:, F16.POS_NEU] + b_times[:, None] * b_x1[:, F16.VT, None] * b_velangle_1 * z_flip
        a = 0.1
        b_x_infront = b_x.at[:, F16.POS_NEU].set((1 - a) * b_x1_point + a * b_x0[:, F16.POS_NEU])

        return b_x_infront

    def sample_train_x0_point(self, key: PRNGKey, n_sample: int):
        key0, key_smooth, key_point_t = jr.split(key, 3)

        x_lb, x_ub = self.train_bounds()
        b_x = jr.uniform(key0, (n_sample, self.nx), minval=x_lb, maxval=x_ub)
        b_x0, b_x1 = b_x[:, : F16.NX], b_x[:, F16.NX :]

        # For this, smooth out the initial state of plane0.
        b_a = jr.uniform(key_smooth, (n_sample,), minval=0.0, maxval=0.5)
        x = self.eq_state()[: F16.NX]
        b_alpha_sm = (1 - b_a) * x[F16.ALPHA] + b_a * b_x[:, F16.ALPHA]
        b_beta_sm = (1 - b_a) * x[F16.BETA] + b_a * b_x[:, F16.BETA]
        b_theta_sm = (1 - b_a) * x[F16.THETA] + b_a * b_x[:, F16.THETA]
        b_p_sm = (1 - b_a) * x[F16.P] + b_a * b_x0[:, F16.P]
        b_q_sm = (1 - b_a) * x[F16.Q] + b_a * b_x0[:, F16.Q]
        b_r_sm = (1 - b_a) * x[F16.R] + b_a * b_x0[:, F16.R]
        b_h_sm = (1 - b_a) * b_x1[:, F16.H] + b_a * b_x0[:, F16.H]
        b_nzint_sm = (1 - b_a) * x[F16.NZINT] + b_a * b_x[:, F16.NZINT]
        b_x_sm = b_x.at[:, F16.ALPHA].set(b_alpha_sm).at[:, F16.BETA].set(b_beta_sm).at[:, F16.THETA].set(b_theta_sm)
        b_x_sm = b_x_sm.at[:, F16.P].set(b_p_sm).at[:, F16.Q].set(b_q_sm).at[:, F16.R].set(b_r_sm)
        b_x_sm = b_x_sm.at[:, F16.H].set(b_h_sm).at[:, F16.NZINT].set(b_nzint_sm)
        b_x = b_x_sm

        b_x0, b_x1 = b_x[:, : F16.NX], b_x[:, F16.NX :]

        # Sample intersection times, then point the plane at those points.
        b_velangle_1 = jax_vmap(compute_f16_vel_angles)(b_x1)

        b_t_point = jr.uniform(key_point_t, (n_sample,), minval=-1.0, maxval=5.0)
        b_x1_point = b_x1[:, F16.POS_NEU] + b_t_point[:, None] * b_x1[:, F16.VT, None] * b_velangle_1
        b_pos1mpos0 = b_x1_point[:, :2] - b_x0[:, F16.POS2D_NED]

        b_yaw_point = jnp.arctan2(b_pos1mpos0[:, 1], b_pos1mpos0[:, 0])
        b_yaw_noise = b_x0[:, F16.PSI] * 0.1
        b_yaw = b_yaw_point + b_yaw_noise
        b_x = b_x.at[:, self.PSI0].set(b_yaw)

        return b_x

    def sample_train_x0_diffangle(self, key: PRNGKey, n_sample: int):
        key0, key_smooth, key_pe, key_pn, key_pn2, key_narrow, key_yaw0, key_yaw1 = jr.split(key, 8)

        # For this one, heavily smooth x0, and force it to face east (pi/2)
        x_lb, x_ub = self.train_bounds()
        b_x = jr.uniform(key0, (n_sample, self.nx), minval=x_lb, maxval=x_ub)
        b_x0, b_x1 = b_x[:, : F16.NX], b_x[:, F16.NX :]

        b_a = jr.uniform(key_smooth, (n_sample,), minval=0.0, maxval=0.1)
        x = self.eq_state()[: F16.NX]
        b_x = b_x.at[:, : F16.NX].set((1 - b_a[:, None]) * x + b_a[:, None] * b_x0)

        b_pe = jr.uniform(key_pe, (n_sample,), minval=-3_500, maxval=0.0)
        b_x = b_x.at[:, self.PE0].set(b_pe)
        b_x = b_x.at[:, self.PSI0].set(np.pi / 2)

        #####################################################################3
        # Random PN and yaw for plane1.
        b_pn = 400 * jr.normal(key_pn, (n_sample,))
        b_pn = b_pn + jr.uniform(key_pn2, (n_sample,), minval=-200.0, maxval=200.0)
        b_x = b_x.at[:, self.PN1].set(b_pn)

        p_use_narrow = 0.8

        psimax = np.pi / 2
        b_yaw0 = -(np.pi / 2 + jnp.sign(b_pn) * jr.uniform(key_yaw0, (n_sample,), minval=0.0, maxval=0.3 * psimax))
        b_yaw1 = -(np.pi / 2 + jnp.sign(b_pn) * jr.uniform(key_yaw1, (n_sample,), minval=0.2 * psimax, maxval=psimax))
        b_use_narrow = jr.bernoulli(key_narrow, p_use_narrow, (n_sample,))
        # Only use narrow if abs(PN) is larger than 500.
        b_use_narrow = jnp.logical_and(b_use_narrow, jnp.abs(b_pn) > 500)

        b_yaw = jnp.where(b_use_narrow, b_yaw1, b_yaw0)
        b_x = b_x.at[:, self.PSI1].set(b_yaw)

        return b_x

    def sample_train_x0(self, key: PRNGKey, n_sample: int) -> BState:
        frac_rand = 0.3
        frac_infront = 0.3
        frac_point = 0.3
        # frac_diffangle = 0.1

        n_rand = int(frac_rand * n_sample)
        n_infront = int(frac_infront * n_sample)
        n_point = int(frac_point * n_sample)
        n_diffangle = n_sample - (n_rand + n_infront + n_point)

        key_rand, key_infront, key_point, key_diffangle = jr.split(key, 4)
        b_x_rand = self.sample_train_x0_random(key_rand, n_rand)
        b_x_infront = self.sample_train_x0_infront(key_infront, n_infront)
        b_x_point = self.sample_train_x0_point(key_point, n_point)
        b_x_diffangle = self.sample_train_x0_diffangle(key_diffangle, n_diffangle)

        b_x = jnp.concatenate([b_x_rand, b_x_infront, b_x_point, b_x_diffangle], axis=0)
        assert b_x.shape == (n_sample, self.nx)
        return b_x

    def contour_bounds(self) -> Float[Arr, "2 nx"]:
        return self.train_bounds()

    def get_metric_x0(self, n_pts: int = 256) -> BState:
        return super().get_metric_x0(n_pts=n_pts)

    def get_plot_x0(self, setup_idx: int = 0, n_pts: int = 8) -> BState:
        return super().get_plot_x0(setup_idx, n_pts=n_pts)

    def _phase2d_setups(self) -> list[Task.Phase2DSetup]:
        # Top down 2d plots for a variety of initial rolls for the other plane?
        get2d = Task.mk_get2d([self.PE0, self.PN0])
        return [
            Task.Phase2DSetup("td2d", self.plot_pos2d, get2d),
            Task.Phase2DSetup("eastup", self.plot_eastup, Task.mk_get2d([self.PE0, self.H0])),
            Task.Phase2DSetup("altpitch", self.plot_altpitch, Task.mk_get2d([self.H0, self.THETA0])),
            ##################
            Task.Phase2DSetup("td2d_tilt", ft.partial(self.plot_pos2d, plane1_pos=(0, 1000)), get2d),
        ]

    def get_contour_x0(self, setup: int = 0, n_pts: int = 80):
        setups = self.phase2d_setups()
        if setups[setup].plot_name == "td2d_tilt":
            with jax.ensure_compile_time_eval():
                phase2d_setup = self.phase2d_setups()[setup]
                contour_bounds = self.contour_bounds()
                contour_bounds[:, self.PN0] = np.array([-1_000, 1_000])

                idxs = [self.PE0, self.PN0]
                nom_state = self.nominal_val_state()
                nom_state[self.PE1] = 0.0
                nom_state[self.PN1] = 1_000.0
                nom_state[self.PSI1] = -0.75 * np.pi
                bb_Xs, bb_Ys, bb_x0 = get_mesh_np(contour_bounds, idxs, n_pts, n_pts, nom_state)
            return bb_x0, bb_Xs, bb_Ys
        else:
            return super().get_contour_x0(setup, n_pts)

    def plot_pos2d(self, ax: plt.Axes, plane1_pos=(0, 0)):
        train_bounds = self.train_bounds()
        # PLOT_XMIN, PLOT_XMAX = plot_bounds[:, self.PE0]
        # PLOT_YMIN, PLOT_YMAX = plot_bounds[:, self.PN0]
        # PLOT_XMIN, PLOT_XMAX = -4_000, 4_000
        # PLOT_YMIN, PLOT_YMAX = -4_000, 4_000
        PLOT_XMIN, PLOT_XMAX = -3_100, 1_700
        PLOT_YMIN, PLOT_YMAX = -1_000, 1_000
        ax.set(xlim=(PLOT_XMIN, PLOT_XMAX), ylim=(PLOT_YMIN, PLOT_YMAX))
        ax.set(aspect="equal")
        ax.set(xlabel=r"East (ft)", ylabel=r"North (ft)")

        # Shade the training bounds.
        bounds_opts = dict(color="C0", alpha=0.2)
        ax.axhspan(PLOT_YMIN, train_bounds[0, self.PN0], **bounds_opts)
        ax.axhspan(train_bounds[1, self.PN0], PLOT_YMAX, **bounds_opts)

        ax.axvspan(PLOT_XMIN, train_bounds[0, self.PE0], **bounds_opts)
        ax.axvspan(train_bounds[1, self.PE0], PLOT_XMAX, **bounds_opts)

        # plane1 is facing West at (0, 0).

        # Plot the circle.
        plane1_pos = np.array(plane1_pos)
        circ_offset = np.array([20.0, 0.0])
        circ = plt.Circle(plane1_pos + circ_offset, 42.0 + 30.0, **PlotStyle.obs_region)
        ax.add_patch(circ)

    def plot_eastup(self, ax: plt.Axes):
        plot_bounds = self.train_bounds()
        PLOT_XMIN, PLOT_XMAX = plot_bounds[:, self.PE0]
        PLOT_YMIN, PLOT_YMAX = plot_bounds[:, self.H0]
        ax.set(xlim=(PLOT_XMIN, PLOT_XMAX), ylim=(PLOT_YMIN, PLOT_YMAX))
        ax.set(aspect="equal")
        ax.set(xlabel=r"East (ft)", ylabel=r"Alt (ft)")

        # plane1 is facing West at (0, 0).

        # Plot the circle.
        nom_h = self.nominal_val_state()[self.H1]
        circ = plt.Circle((20, nom_h), 42.0 + 30.0, **PlotStyle.obs_region)
        ax.add_patch(circ)

    def nom_pol_pid(self, state: State):
        nom_alt = self.nominal_val_state()[F16.H]
        pid_controller = F16PIDController(nom_alt)
        full_control = pid_controller.get_control(state)
        control = full_control[: self.NU]
        return jnp.clip(control, self.u_min, self.u_max)

    def nom_pol_N0_pid(self, state: State) -> Control:
        full_state = state
        state = full_state[: F16.NX]

        nom_alt = self.nominal_val_state()[F16.H]
        pid_controller = F16N0PIDController(nom_alt)
        full_control = pid_controller.get_control(state)
        control = full_control[: self.NU]
        return jnp.clip(control, self.u_min, self.u_max)

    def nom_pol_N0_pid_old(self, state: State) -> Control:
        full_state = state
        state = full_state[: F16.NX]

        vt, alpha = state[S.VT], state[S.ALPHA]
        theta = state[S.THETA]
        h = state[S.ALT]

        kp_alt = 0.025
        kgamma_alt = 25

        kp_psi_pn = 0.5 * 0.004
        kd_psi_pn = 0.5 * 0.010

        kp_psi = 5.0
        kd_psi = 0.5

        kp_phi = 0.75
        kd_phi = 0.5

        max_bank_deg = np.deg2rad(65)

        # 1: PD on Nz to reach target altitude.
        #    Path angle (rad)
        gamma = theta - alpha

        alt_setpoint = self.eq_state()[self.H0]

        h_err = alt_setpoint - h
        Nz = kp_alt * h_err
        Nz = jnp.clip(Nz, -1.0, 6.0)

        north_vel = state[F16.VT] * compute_f16_vel_angles(state)[0]

        Nz = Nz - kgamma_alt * gamma
        nz_cmd = jnp.clip(Nz, -1.0, 6.0)

        # 2: PID on yaw to face East (psi = pi/2)
        pn_err = 0.0 - state[F16.PN]

        psi_cmd = np.pi / 2
        psi_err = wrap_to_pi(psi_cmd - state[F16.PSI])
        # Clip the p term.
        phi_pn_contrib = -(jnp.clip(pn_err * kp_psi_pn, -0.3, 0.3) - north_vel * kd_psi_pn)
        phi_pn_contrib = phi_pn_contrib.clip(-max_bank_deg, max_bank_deg)
        phi_cmd = phi_pn_contrib + psi_err * kp_psi - state[F16.R] * kd_psi

        # Clip the phi_cmd to [-pi/2, pi/2]. We don't want to be upside down.
        phi_cmd = jnp.clip(phi_cmd, -max_bank_deg, max_bank_deg)

        # Wrap the error to [-pi, pi].
        # phi_err = wrap_to_pi(phi_cmd - state[F16.PHI])
        phi_err = phi_cmd - state[F16.PHI]
        ps_cmd = phi_err * kp_phi - state[F16.P] * kd_phi

        return jnp.array([nz_cmd, ps_cmd, 0.0]).clip(self.u_min, self.u_max)

    def plot_plane_constraints(self, axes: list[plt.Axes]):
        # Constraints on [ alpha, beta, alt, pitch ]
        assert len(axes) == F16.NX
        axes = [axes[F16.ALPHA], axes[F16.BETA], axes[F16.THETA], axes[F16.H]]
        bounds = np.array([A_BOUNDS, B_BOUNDS, [-np.pi, np.pi], [0.0, self._alt_max]]).T
        plot_boundaries_with_clip(axes, bounds, color="C0")

    def plot_altpitch(self, ax: plt.Axes):
        # x is altitude, y is pitch=theta
        train_bounds = self.train_bounds()
        PLOT_X_MIN, PLOT_X_MAX = train_bounds[:, self.H0]
        PLOT_Y_MIN, PLOT_Y_MAX = train_bounds[:, self.THETA0]
        ax.set(xlim=(PLOT_X_MIN, PLOT_X_MAX), ylim=(PLOT_Y_MIN, PLOT_Y_MAX))

        # Plot the avoid set.
        plot_x_bounds(ax, (0.0, self._alt_max), PlotStyle.obs_region)
        plot_y_bounds(ax, (-1.2, 1.2), PlotStyle.obs_region)
        ax.set(xlabel=r"alt (ft)", ylabel=r"$\theta$ (rad)")

    def plot_alphabeta(self, ax: plt.Axes):
        train_bounds = self.train_bounds()
        PLOT_X_MIN, PLOT_X_MAX = train_bounds[:, self.BETA0]
        PLOT_Y_MIN, PLOT_Y_MAX = train_bounds[:, self.ALPHA0]
        ax.set(xlim=(PLOT_X_MIN, PLOT_X_MAX), ylim=(PLOT_Y_MIN, PLOT_Y_MAX))

        # Plot the avoid set.
        plot_x_bounds(ax, B_BOUNDS, PlotStyle.obs_region)
        plot_y_bounds(ax, A_BOUNDS, PlotStyle.obs_region)
        ax.set(xlabel=r"$\beta$ (rad)", ylabel=r"$\alpha$ (rad)")

    def plot_alphavt(self, ax: plt.Axes):
        # x = alpha, y = vt
        train_bounds = self.train_bounds()
        PLOT_X_MIN, PLOT_X_MAX = train_bounds[:, self.ALPHA0]
        PLOT_Y_MIN, PLOT_Y_MAX = train_bounds[:, self.VT0]
        ax.set(xlim=(PLOT_X_MIN, PLOT_X_MAX), ylim=(PLOT_Y_MIN, PLOT_Y_MAX))

        # Plot the avoid set.
        plot_x_bounds(ax, A_BOUNDS, PlotStyle.obs_region)
        ax.set(xlabel=r"$\alpha$ (rad)", ylabel=r"$v_T$")

    def plot_boundaries(self, axes: list[plt.Axes]):
        axes_bounded = [
            axes[self.ALPHA0],
            axes[self.BETA0],
            axes[self.H0],
            axes[self.THETA0],
            axes[self.PSINT0],
            axes[self.NYRINT0],
        ]
        lb = np.array([A_BOUNDS[0], B_BOUNDS[0], 0.0, -1.2, -2.5, -2.5])
        ub = np.array([A_BOUNDS[1], B_BOUNDS[1], self._alt_max, 1.2, 2.5, 2.5])
        bounds = np.stack([lb, ub], axis=0)
        plot_boundaries(axes_bounded, bounds, color="C0")


A_BOUNDS = np.array([-0.17453292519943295, 0.7853981633974483])
B_BOUNDS = np.array([-0.5235987755982988, 0.5235987755982988])


def get_R_W_B(state: State) -> RotMat3D:
    return rotz(state[S.PSI]) @ roty(state[S.THETA]) @ rotx(state[S.PHI])


def get_R_B_Wnd(state: State) -> RotMat3D:
    return roty(-state[S.ALPHA]) @ rotz(state[S.BETA])


def get_R_W_Wnd(state: State) -> RotMat3D:
    return get_R_W_B(state) @ get_R_B_Wnd(state)


def get_pos_NED(state: State) -> Vec3:
    return jnp.array([state[F16.PN], state[F16.PE], -state[F16.H]])
