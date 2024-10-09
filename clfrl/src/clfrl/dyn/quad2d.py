import jax
import jax.numpy as jnp
import numpy as np
from attrs import define

from clfrl.dyn.dyn_types import HFloat, PolObs, State, VObs
from clfrl.dyn.task import Task
from clfrl.utils.costconstr_utils import poly4_clip_max_flat
from clfrl.utils.small_la import inv33


class Quad2D(Task):
    NQ = 3
    NX = 6
    NU = 2
    PX, PZ, THETA, VX, VZ, OMEGA = range(NX)
    # Thrust right, thrust left.
    TR, TL = range(NU)

    @define
    class Params:
        # Mass of quad.
        mq: float = 0.486
        # Half of wingspan.
        l: float = 0.25
        # Moment of inertia.
        J: float = 0.00383
        # Acceleration due to Gravity.
        g: float = 9.81
        # Linear Drag.
        b_lin: float = 0.0
        # Angular Drag.
        b_ang: float = 0.0

    params: Params

    def __init__(self, params: Params = Params()):
        self.params = params
        self.px_max = 2.0

    @property
    def n_Vobs(self) -> int:
        # [px, pz, sin, cos, vx, vz, omega]
        return 7

    def x_labels(self) -> list[str]:
        return [r"$p_x$", r"$p_z$", r"$\theta$", r"$v_x$", r"$v_z$", r"$\omega$"]

    def u_labels(self) -> list[str]:
        return [r"$u_R$", r"$u_L$"]

    def h_labels(self) -> list[str]:
        return [r"$x_l$", r"$x_r$", "floor"]

    @property
    def h_max(self) -> float:
        # return 0.75
        return 1.0

    def h_components(self, state: State) -> HFloat:
        px, pz, th, vx, vy, w = self.chk_x(state)
        px_lb = px - self.px_max
        px_ub = -(px + self.px_max)
        pz_lb = -pz

        hs = jnp.array([px_lb, px_ub, pz_lb])
        # h <= 1
        hs = poly4_clip_max_flat(hs)
        return hs

    def get_obs(self, state: State) -> tuple[VObs, PolObs]:
        px, pz, th, vx, vy, w = self.chk_x(state)
        sin, cos = jnp.sin(th), jnp.cos(th)
        obs = jnp.array([px, pz, sin, cos, vx, vy, w])
        assert obs.shape == (self.n_Vobs,)
        return obs, obs

    def nominal_controls(self, control_type: str, state: State | None = None) -> jnp.ndarray:
        p = self.params
        if control_type == "hover":
            with jax.ensure_compile_time_eval():
                nom_u = 0.5 * p.mq * p.g
                return jnp.array([nom_u, nom_u])
        if control_type == "keep_altitude":
            assert state is not None
            assert jnp.equal(state[Quad2D.OMEGA], 0.0)
            with jax.ensure_compile_time_eval():
                nom_u = 0.5 * p.mq * p.g / jnp.cos(state[Quad2D.THETA])
                return jnp.array([nom_u, nom_u])
        else:
            raise NotImplementedError("")

    def partial_V(self) -> State:
        p = self.params
        return np.array([0.0, p.mq * p.g, 0.0])

    def f(self, state: State) -> State:
        p = self.params

        with jax.ensure_compile_time_eval():
            f = jnp.zeros(self.NX)
            partial_V = self.partial_V()
            M = self.M()
            M_inv = inv33(M)

        #      [ lin_drag, lin_drag, ang_drag ]
        forces = jnp.array([-p.b_lin * state[self.VX], -p.b_lin * state[self.VZ], -p.b_ang * state[self.OMEGA]])
        f = f.at[self.NQ :].set(M_inv @ (-partial_V + forces))

        # \dot{} = dq.
        f = f.at[: self.NQ].set(state[self.NQ :])
        return f

    def M(self):
        with jax.ensure_compile_time_eval():
            p = self.params
            return np.diag(np.array([p.mq, p.mq, p.J]))

    def G(self, state: State):
        p = self.params
        with jax.ensure_compile_time_eval():
            G = jnp.zeros((Quad2D.NX, Quad2D.NU))
            M = self.M()
            M_inv = inv33(M)

        cos_t, sin_t = jnp.cos(state[Quad2D.THETA]), jnp.sin(state[Quad2D.THETA])
        B = jnp.array([[-sin_t, -sin_t], [cos_t, cos_t], [p.l, -p.l]])

        G = G.at[self.NQ :, :].set(M_inv @ B)
        return G
