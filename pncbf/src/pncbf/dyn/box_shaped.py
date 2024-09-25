import einops as ei
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Float

from pncbf.dyn.dyn_types import BState, Control, Disturb, HFloat, LFloat, PolObs, State, TState, VObs
from pncbf.dyn.task import Task
from pncbf.plotting.phase2d_utils import plot_x_bounds
from pncbf.plotting.plotstyle import PlotStyle
from pncbf.utils.costconstr_utils import add_constr_margin, poly4_clip_max_flat
from pncbf.utils.jax_types import Arr, BoolScalar
from pncbf.utils.jax_utils import jax_vmap
from pncbf.utils.sampling_utils import get_mesh_np


class BoxShaped(Task):
    NX = 2
    NU = 1
    ND = 1

    P, V = range(NX)
    (A,) = range(NU)

    H_TILT_EPS = 0.1

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
        return [r"$p_{lb}$", r"$p_{ub}$"]

    @property
    def l_scale(self) -> LFloat:
        return np.array([0.0])

    def l_components(self, state: State) -> LFloat:
        return jnp.array([0.0])

    @property
    def h_max(self) -> float:
        return 0.75

    def h_components(self, state: State) -> HFloat:
        self.chk_x(state)
        p, v = state
        h_p_ub = (p - 1.0) / self.H_TILT_EPS
        h_p_lb = -(p + 1.0) / (2.0 - self.H_TILT_EPS)
        # h <= 1
        hs = poly4_clip_max_flat(jnp.array([h_p_lb, h_p_ub]))
        # Make sure h is bounded.
        hs = jnp.clip(0.25 * hs, a_max=self.h_max - 0.5)
        # Add a margin.
        return add_constr_margin(hs, 1.0)

    def is_stable(self, T_state: TState) -> BoolScalar:
        # If the position and velocity haven't changed significantly in the last 5 steps.
        max_state_diff = jnp.abs(T_state[-5:] - T_state[-5]).max(axis=0)
        self.chk_x(max_state_diff)
        pos_same = max_state_diff[0] < 0.02
        vel_same = max_state_diff[1] < 0.02
        return pos_same & vel_same

    def get_obs(self, state: State) -> tuple[VObs, PolObs]:
        # Prevent obs from exploding if the state explodes.
        obs = state.clip(-5.0, 5.0)
        return obs, obs

    @property
    def dt(self):
        return 0.1

    def f(self, state: State) -> State:
        p, v = self.chk_x(state)
        return jnp.array([v, 0.0])

    def G(self, state: State):
        GT = np.array([[0, 1]])
        G = GT.T
        return G

    def step(self, state: State, control: Control, disturb: Disturb = None) -> State:
        """Disturb enters through control channel, and has half the control authority of the control."""
        dt = self.dt
        p, v = state
        noisy_control = control
        if disturb is not None:
            noisy_control = control + 0.5 * disturb
        (a,) = noisy_control

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
        return np.array([[-0.75, 0.6]])

    def in_ci_approx(self, state: State) -> BoolScalar:
        x, v = state
        in_right = x <= 1 - jnp.maximum(v, 0) ** 2 / 2
        in_left = x >= jnp.minimum(v, 0) ** 2 / 2 - 1
        return in_left & in_right

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
        vs = np.linspace(-1.5, 1.5)
        xs = np.minimum(vs, 0.0) ** 2 / 2 - 1
        all_xs += [xs]
        all_vs += [vs]

        vs = np.linspace(-1.5, 1.5)[::-1]
        xs = 1 - np.maximum(vs, 0.0) ** 2 / 2
        all_xs += [xs]
        all_vs += [vs]

        all_xs.append(all_xs[0])
        all_vs.append(all_vs[0])

        xs, vs = np.concatenate(all_xs, axis=0), np.concatenate(all_vs, axis=0)
        ax.plot(xs, vs, **PlotStyle.ci_line)

        # 2: Plot the avoid set.
        plot_x_bounds(ax, (-1.0, 1.0), PlotStyle.obs_region)

        # 4: Plot the switching surface for the unconstrained.
        vs = np.linspace(PLOT_YMIN, PLOT_YMAX)
        xs = (1 - self.H_TILT_EPS) - np.maximum(vs, 0.0) ** 2 / (2 * 1.0)
        ax.plot(xs, vs, **PlotStyle.switch_line)

        # 5: Plot the switching surface for adversarial.
        vs = np.linspace(-np.sqrt(2), np.sqrt(2))
        xs = 1 - np.maximum(vs, 0.0) ** 2 / (2 * 0.5)
        ax.plot(xs, vs, **PlotStyle.switch_line)
        xs = np.minimum(vs, 0.0) ** 2 / (2 * 0.5) - 1
        ax.plot(xs, vs, **PlotStyle.switch_line)
