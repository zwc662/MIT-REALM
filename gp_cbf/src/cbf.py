import jax.numpy as jnp
import jax
import einops as ei

from abc import ABC

from typing import Optional, Tuple
from jaxproxqp.jaxproxqp import JaxProxQP

from gcbfplus.utils.typing import Action, Params, PRNGKey, Array, State, Done, Reward
from gcbfplus.utils.graph import GraphsTuple
from gcbfplus.utils.utils import mask2index
from gcbfplus.trainer.data import Rollout
from gcbfplus.utils import get_pwise_cbf_fn

from .agent import AgentController
from .env import Environment



class CBF(ABC):
    def compute_gae_fn(
            self, values: Array, rewards: Reward, dones: Done, next_values: Array, gamma: float, gae_lambda: float
    ) -> Tuple[Array, Array]:
        """
        Compute generalized advantage estimation.
        """
        deltas = rewards + gamma * next_values * (1 - dones) - values
        gaes = deltas

        def scan_fn(gae, inp):
            delta, done = inp
            gae_prev = delta + gamma * gae_lambda * (1 - done) * gae
            return gae_prev, gae_prev

        _, gaes_prev = jax.lax.scan(scan_fn, gaes[-1], (deltas[:-1], dones[:-1]), reverse=True)
        gaes = jnp.concatenate([gaes_prev, gaes[-1, None]], axis=0)

        return gaes + values, (gaes - gaes.mean()) / (gaes.std() + 1e-8)


    def compute_gae(
            self, values: Array, rewards: Reward, dones: Done, next_values: Array, gamma: float, gae_lambda: float
    ) -> Tuple[Array, Array]:
        return jax.vmap(ft.partial(compute_gae_fn, gamma=gamma, gae_lambda=gae_lambda))(values, rewards, dones, next_values)


    def pwise_cbf_single_integrator_(
            self, pos: Array, agent_idx: int, o_obs_pos: Array, a_pos: Array, r: float, k: int):
        n_agent = len(a_pos)

        all_obs_pos = jnp.concatenate([a_pos, o_obs_pos], axis=0)

        # Only consider the k obstacles.
        o_dist_sq = ((pos - all_obs_pos) ** 2).sum(axis=-1)
        # Remove self collisions
        o_dist_sq = o_dist_sq.at[agent_idx].set(1e2)

        # Take the k closest obstacles.
        k_idx = jnp.argsort(o_dist_sq)[:k]
        k_dist_sq = o_dist_sq[k_idx]
        # Take radius into account. Add some epsilon for qp solver error.
        k_dist_sq = k_dist_sq - 4 * (1.01 * r) ** 2

        k_h0 = k_dist_sq
        k_isobs = k_idx >= n_agent

        return k_h0, k_isobs


    def pwise_cbf_single_integrator(
            self, graph: GraphsTuple, r: float, n_agent: int, n_rays: int, k: int):
        # (n_agents, 2)
        a_states = graph.type_states(type_idx=0, n_type=n_agent)
        # (n_obs, 2)
        obs_states = graph.type_states(type_idx=2, n_type=n_agent * n_rays)
        a_obs_states = ei.rearrange(obs_states, "(n_agent n_ray) d -> n_agent n_ray d", n_agent=n_agent)

        agent_idx = jnp.arange(n_agent)
        fn = jax.vmap(ft.partial(pwise_cbf_single_integrator_, r=r, k=k), in_axes=(0, 0, 0, None))
        ak_h0, ak_isobs = fn(a_states, agent_idx, a_obs_states, a_states)
        return ak_h0, ak_isobs


    def pwise_cbf_double_integrator_(
            self, state: Array, agent_idx: int, o_obs_state: Array, a_state: Array, r: float, k: int):
        n_agent = len(a_state)

        pos = state[:2]
        all_obs_state = jnp.concatenate([a_state, o_obs_state], axis=0)
        all_obs_pos = all_obs_state[:, :2]
        del o_obs_state

        # Only consider the k closest obstacles.
        o_dist_sq = ((pos - all_obs_pos) ** 2).sum(axis=-1)
        # Remove self collisions
        o_dist_sq = o_dist_sq.at[agent_idx].set(1e2)
        # Take the k closest obstacles.
        k_idx = jnp.argsort(o_dist_sq)[:k]
        k_dist_sq = o_dist_sq[k_idx]
        # Take radius into account.
        k_dist_sq = k_dist_sq - 4 * r ** 2

        k_h0 = k_dist_sq
        assert k_h0.shape == (k,)

        k_xdiff = state[:2] - all_obs_state[k_idx][:, :2]
        k_vdiff = state[2:] - all_obs_state[k_idx][:, 2:]
        assert k_xdiff.shape == k_vdiff.shape == (k, 2)

        k_h0_dot = 2 * (k_xdiff * k_vdiff).sum(axis=-1)
        assert k_h0_dot.shape == (k,)

        k_h1 = k_h0_dot + 10.0 * k_h0

        k_isobs = k_idx >= n_agent

        return k_h1, k_isobs


    def pwise_cbf_double_integrator(
            self, graph: GraphsTuple, r: float, n_agent: int, n_rays: int, k: int):
        # (n_agents, 4)
        a_states = graph.type_states(type_idx=0, n_type=n_agent)
        # (n_obs, 4)
        obs_states = graph.type_states(type_idx=2, n_type=n_agent * n_rays)
        a_obs_states = ei.rearrange(obs_states, "(n_agent n_ray) d -> n_agent n_ray d", n_agent=n_agent)

        agent_idx = jnp.arange(n_agent)
        fn = jax.vmap(ft.partial(pwise_cbf_double_integrator_, r=r, k=k), in_axes=(0, 0, 0, None))
        ak_h0, ak_isobs = fn(a_states, agent_idx, a_obs_states, a_states)
        return ak_h0, ak_isobs


    def pwise_cbf_dubins_car_(
            self, state: Array, agent_idx: int, o_obs_state: Array, a_state: Array, r: float, k: int):
        n_agent = len(a_state)
        n_obs = len(o_obs_state)

        pos = state[:2]
        vel = state[3] * jnp.array([jnp.cos(state[2]), jnp.sin(state[2])])
        assert vel.shape == (2,)

        agent_vel = a_state[:, 3, None] * jnp.stack([jnp.cos(a_state[:, 2]), jnp.sin(a_state[:, 2])], axis=-1)
        assert agent_vel.shape == (n_agent, 2)

        all_obs_pos = jnp.concatenate([a_state[:, :2], o_obs_state[:, :2]], axis=0)
        all_obs_vel = jnp.concatenate([agent_vel, jnp.zeros((n_obs, 2))], axis=0)
        del o_obs_state

        # Only consider the k closest obstacles.
        o_dist_sq = ((pos - all_obs_pos) ** 2).sum(axis=-1)
        # Remove self collisions
        o_dist_sq = o_dist_sq.at[agent_idx].set(1e2)
        # Take the k closest obstacles.
        k_idx = jnp.argsort(o_dist_sq)[:k]
        k_dist_sq = o_dist_sq[k_idx]
        # Take radius into account. Add some epsilon for qp solver error.
        k_dist_sq = k_dist_sq - 4 * r ** 2

        k_h0 = k_dist_sq
        assert k_h0.shape == (k,)

        k_xdiff = state[:2] - all_obs_pos[k_idx]
        k_vdiff = agent_vel[agent_idx] - all_obs_vel[k_idx]
        assert k_xdiff.shape == k_vdiff.shape == (k, 2)

        k_h0_dot = 2 * (k_xdiff * k_vdiff).sum(axis=-1)
        assert k_h0_dot.shape == (k,)

        k_h1 = k_h0_dot + 5.0 * k_h0

        k_isobs = k_idx >= n_agent

        return k_h1, k_isobs


    def pwise_cbf_dubins_car(
            self, graph: GraphsTuple, r: float, n_agent: int, n_rays: int, k: int):
        # (n_agents, 4)
        a_states = graph.type_states(type_idx=0, n_type=n_agent)
        # (n_obs, 4)
        obs_states = graph.type_states(type_idx=2, n_type=n_agent * n_rays)
        a_obs_states = ei.rearrange(obs_states, "(n_agent n_ray) d -> n_agent n_ray d", n_agent=n_agent)

        agent_idx = jnp.arange(n_agent)
        fn = jax.vmap(ft.partial(pwise_cbf_dubins_car_, r=r, k=k), in_axes=(0, 0, 0, None))
        ak_h0, ak_isobs = fn(a_states, agent_idx, a_obs_states, a_states)
        return ak_h0, ak_isobs


    def pwise_cbf_crazyflie_(
            self, state: Array, agent_idx: int, o_obs_state: Array, a_state: Array, r: float, k: int):
        # state: ( 12, )
        n_agent = len(a_state)

        pos = state[:3]
        all_obs_state = jnp.concatenate([a_state, o_obs_state], axis=0)
        all_obs_pos = all_obs_state[:, :3]
        del o_obs_state

        # Only consider the k closest obstacles.
        o_dist_sq = ((pos - all_obs_pos) ** 2).sum(axis=-1)
        # Remove self collisions
        o_dist_sq = o_dist_sq.at[agent_idx].set(1e2)
        # Take the k closest obstacles.
        k_idx = jnp.argsort(o_dist_sq)[:k]
        k_dist_sq = o_dist_sq[k_idx]
        # Take radius into account. Add some epsilon for qp solver error.
        k_dist_sq = k_dist_sq - 4 * (1.01 * r) ** 2

        k_h0 = k_dist_sq
        assert k_h0.shape == (k,)

        # all_obs_state = all_obs_state[k_idx]

        def crazyflie_f_(x: Array) -> Array:
            Ixx, Iyy, Izz = CrazyFlie.PARAMS["Ixx"], CrazyFlie.PARAMS["Iyy"], CrazyFlie.PARAMS["Izz"]
            I = jnp.array([Ixx, Iyy, Izz])
            # roll, pitch, yaw
            phi, theta, psi = x[CrazyFlie.PHI], x[CrazyFlie.THETA], x[CrazyFlie.PSI]
            c_phi, s_phi = jnp.cos(phi), jnp.sin(phi)
            c_th, s_th = jnp.cos(theta), jnp.sin(theta)
            c_psi, s_psi = jnp.cos(psi), jnp.sin(psi)
            t_th = jnp.tan(theta)

            u, v, w = x[CrazyFlie.U], x[CrazyFlie.V], x[CrazyFlie.W]
            uvw = jnp.array([u, v, w])

            p, q, r = x[CrazyFlie.P], x[CrazyFlie.Q], x[CrazyFlie.R]
            pqr = jnp.array([p, q, r])

            # Linear velocity
            R_W_cf = jnp.array(
                [
                    [c_psi * c_th, c_psi * s_th * s_phi - s_psi * c_phi, c_psi * s_th * c_phi + s_psi * s_phi],
                    [s_psi * c_th, s_psi * s_th * s_phi + c_psi * c_phi, s_psi * s_th * c_phi - c_psi * s_phi],
                    [-s_th, c_th * s_phi, c_th * c_phi],
                ]
            )
            v_Wcf_cf = jnp.array([u, v, w])
            v_Wcf_W = R_W_cf @ v_Wcf_cf
            assert v_Wcf_W.shape == (3,)

            # Euler angle dynamics.
            mat = jnp.array(
                [
                    [0, s_phi / c_th, c_phi / c_th],
                    [0, c_phi, -s_phi],
                    [1, s_phi * t_th, c_phi * t_th],
                ]
            )
            deuler_rpy = mat @ pqr
            deuler_ypr = deuler_rpy[::-1]

            # Body frame linear acceleration.
            acc_cf_g = -R_W_cf[2, :] * 9.81
            acc_cf = -jnp.cross(pqr, uvw) + acc_cf_g

            # Body frame angular acceleration.
            pqr_dot = -jnp.cross(pqr, I * pqr) / I
            rpq_dot = pqr_dot[::-1]
            assert pqr_dot.shape == (3,)

            x_dot = jnp.concatenate([v_Wcf_W, deuler_ypr, acc_cf, rpq_dot], axis=0)
            return x_dot

        def h0(x, obs_x):
            k_xdiff = x[:3] - obs_x[:, :3]
            dist = jnp.square(k_xdiff).sum(axis=-1)
            dist = dist.at[agent_idx].set(1e2)
            return dist[k_idx] - 4 * r ** 2  # (k,)

        def h1(x, obs_x):
            x_dot = crazyflie_f_(x)  # (nx,)
            obs_x_dot = jax.vmap(crazyflie_f_)(obs_x)  # (k, nx)

            h0_x = jax.jacfwd(h0, argnums=0)(x, obs_x)  # (k, nx)
            h0_obs_x = jax.jacfwd(h0, argnums=1)(x, obs_x)  # (k, k, nx)
            h0_dot = ei.einsum(h0_x, x_dot, 'k nx, nx -> k') + \
                    ei.einsum(h0_obs_x, obs_x_dot, 'k1 k2 nx, k2 nx -> k1')  # (k,)
            return h0_dot + 30.0 * h0(x, obs_x)

        def h2(x, obs_x):
            x_dot = crazyflie_f_(x)
            obs_x_dot = jax.vmap(crazyflie_f_)(obs_x)
            h1_x = jax.jacfwd(h1, argnums=0)(x, obs_x)  # (k, nx)
            h1_obs_x = jax.jacfwd(h1, argnums=1)(x, obs_x)  # (k, k, nx)
            h1_dot = ei.einsum(h1_x, x_dot, 'k nx, nx -> k') + \
                    ei.einsum(h1_obs_x, obs_x_dot, 'k1 k2 nx, k2 nx -> k1')  # (k,)
            return h1_dot + 50.0 * h1(x, obs_x)

        k_h2 = h2(state, all_obs_state)
        assert k_h2.shape == (k,)

        k_isobs = k_idx >= n_agent

        return k_h2, k_isobs


    def pwise_cbf_crazyflie(
            self, graph: GraphsTuple, r: float, n_agent: int, n_rays: int, k: int):
        # (n_agents, 4)
        a_states = graph.type_states(type_idx=0, n_type=n_agent)
        # (n_obs, 4)
        obs_states = graph.type_states(type_idx=2, n_type=n_agent * n_rays)
        a_obs_states = ei.rearrange(obs_states, "(n_agent n_ray) d -> n_agent n_ray d", n_agent=n_agent)

        agent_idx = jnp.arange(n_agent)
        fn = jax.vmap(ft.partial(pwise_cbf_crazyflie_, r=r, k=k), in_axes=(0, 0, 0, None))
        ak_h0, ak_isobs = fn(a_states, agent_idx, a_obs_states, a_states)
        return ak_h0, ak_isobs


    def pwise_cbf_linear_drone_(
            self, state: Array, agent_idx: int, o_obs_state: Array, a_state: Array, r: float, k: int):
        # state: ( 6, )
        n_agent = len(a_state)

        pos = state[:3]
        all_obs_state = jnp.concatenate([a_state, o_obs_state], axis=0)
        all_obs_pos = all_obs_state[:, :3]
        del o_obs_state

        # Only consider the k closest obstacles.
        o_dist_sq = ((pos - all_obs_pos) ** 2).sum(axis=-1)
        # Remove self collisions
        o_dist_sq = o_dist_sq.at[agent_idx].set(1e2)
        # Take the k closest obstacles.
        k_idx = jnp.argsort(o_dist_sq)[:k]
        k_dist_sq = o_dist_sq[k_idx]
        # Take radius into account. Add some epsilon for qp solver error.
        k_dist_sq = k_dist_sq - 4 * (1.01 * r) ** 2

        k_h0 = k_dist_sq
        assert k_h0.shape == (k,)

        k_xdiff = state[:3] - all_obs_state[k_idx][:, :3]
        k_vdiff = state[3:6] - all_obs_state[k_idx][:, 3:6]
        assert k_xdiff.shape == k_vdiff.shape == (k, 3)

        k_h0_dot = 2 * (k_xdiff * k_vdiff).sum(axis=-1)
        assert k_h0_dot.shape == (k,)

        k_h1 = k_h0_dot + 3.0 * k_h0

        k_isobs = k_idx >= n_agent

        return k_h1, k_isobs


    def pwise_cbf_linear_drone(
            self, graph: GraphsTuple, r: float, n_agent: int, n_rays: int, k: int):
        # (n_agents, 4)
        a_states = graph.type_states(type_idx=0, n_type=n_agent)
        # (n_obs, 4)
        obs_states = graph.type_states(type_idx=2, n_type=n_agent * n_rays)
        a_obs_states = ei.rearrange(obs_states, "(n_agent n_ray) d -> n_agent n_ray d", n_agent=n_agent)

        agent_idx = jnp.arange(n_agent)
        fn = jax.vmap(ft.partial(pwise_cbf_linear_drone_, r=r, k=k), in_axes=(0, 0, 0, None))
        ak_h0, ak_isobs = fn(a_states, agent_idx, a_obs_states, a_states)
        return ak_h0, ak_isobs


    def pwise_cbf_cfhl_(
            self, state: Array, agent_idx: int, o_obs_state: Array, a_state: Array, r: float, k: int):
        # state: ( 6, )
        n_agent = len(a_state)

        pos = state[:3]
        all_obs_state = jnp.concatenate([a_state, o_obs_state], axis=0)
        all_obs_pos = all_obs_state[:, :3]
        del o_obs_state

        # Only consider the k closest obstacles.
        o_dist_sq = ((pos - all_obs_pos) ** 2).sum(axis=-1)
        # Remove self collisions
        o_dist_sq = o_dist_sq.at[agent_idx].set(1e2)
        # Take the k closest obstacles.
        k_idx = jnp.argsort(o_dist_sq)[:k]
        k_dist_sq = o_dist_sq[k_idx]
        # Take radius into account. Add some epsilon for qp solver error.
        k_dist_sq = k_dist_sq - 4 * (1.01 * r) ** 2

        k_h0 = k_dist_sq
        assert k_h0.shape == (k,)

        k_xdiff = state[:3] - all_obs_state[k_idx][:, :3]

        def get_v_single_(x):
            u = x[CrazyFlie.U]
            v = x[CrazyFlie.V]
            w = x[CrazyFlie.W]

            R_W_cf = CrazyFlie.get_rotation_mat(x)
            v_Wcf_cf = jnp.array([u, v, w])
            v_Wcf_W = R_W_cf @ v_Wcf_cf  # world frame velocity
            return v_Wcf_W

        k_vdiff = get_v_single_(state) - jax.vmap(get_v_single_)(all_obs_state[k_idx])

        assert k_xdiff.shape == k_vdiff.shape == (k, 3)

        k_h0_dot = 2 * (k_xdiff * k_vdiff).sum(axis=-1)
        assert k_h0_dot.shape == (k,)

        k_h1 = k_h0_dot + 3 * k_h0

        k_isobs = k_idx >= n_agent

        return k_h1, k_isobs


    def pwise_cbf_cfhl(
            self, graph: GraphsTuple, r: float, n_agent: int, n_rays: int, k: int):
        # (n_agents, 4)
        a_states = graph.type_states(type_idx=0, n_type=n_agent)
        # (n_obs, 4)
        obs_states = graph.type_states(type_idx=2, n_type=n_agent * n_rays)
        a_obs_states = ei.rearrange(obs_states, "(n_agent n_ray) d -> n_agent n_ray d", n_agent=n_agent)

        agent_idx = jnp.arange(n_agent)
        fn = jax.vmap(ft.partial(pwise_cbf_cfhl_, r=r, k=k), in_axes=(0, 0, 0, None))
        ak_h0, ak_isobs = fn(a_states, agent_idx, a_obs_states, a_states)
        return ak_h0, ak_isobs

    def get_pwise_cbf_fn(self, env: Environment, k: int = 3):
        if isinstance(env, SingleIntegrator):
            n_agent = env.num_agents
            n_rays = env.params["n_rays"]
            r = env.params["car_radius"]
            return ft.partial(pwise_cbf_single_integrator, r=r, n_agent=n_agent, n_rays=n_rays, k=k)
        elif isinstance(env, DoubleIntegrator):
            n_agent = env.num_agents
            n_rays = env.params["n_rays"]
            r = env.params["car_radius"]
            return ft.partial(pwise_cbf_double_integrator, r=r, n_agent=n_agent, n_rays=n_rays, k=k)
        elif isinstance(env, DubinsCar):
            r = env.params["car_radius"]
            n_agent = env.num_agents
            n_rays = env.params["n_rays"]
            return ft.partial(pwise_cbf_dubins_car, r=r, n_agent=n_agent, n_rays=n_rays, k=k)
        elif isinstance(env, CrazyFlie):
            return ft.partial(pwise_cbf_crazyflie, r=env.params["drone_radius"], n_agent=env.num_agents, n_rays=env.n_rays,
                            k=k)
        elif isinstance(env, LinearDrone):
            return ft.partial(pwise_cbf_linear_drone, r=env.params["drone_radius"], n_agent=env.num_agents,
                            n_rays=env.n_rays, k=k)
        elif isinstance(env, CrazyFlie):
            return ft.partial(pwise_cbf_cfhl, r=env.params["drone_radius"], n_agent=env.num_agents,
                            n_rays=env.n_rays, k=k)

        raise NotImplementedError("")
        


class QP_CBF(CBF):

    def __init__(
            self,
            env: Environment, 
            state_dim: int,
            action_dim: int, 
            alpha: float = 1.0,
            **kwargs
    ):
        super(QP_CBF, self).__init__(
            state_dim=state_dim
            action_dim=action_dim, 
        )

        self.alpha = alpha
        self.k = 3
        self.cbf = get_pwise_cbf_fn(env, self.k)

    @property
    def config(self) -> dict:
        return {
            'alpha': self.alpha,
        }

    @property
    def actor_params(self) -> Params:
        raise NotImplementedError

    def step(self, graph: GraphsTuple, key: PRNGKey, params: Optional[Params] = None) -> Tuple[Action, Array]:
        raise NotImplementedError

    def get_cbf(self, graph: GraphsTuple) -> Array:
        return self.cbf(graph)[0]

    def update(self, rollout: Rollout, step: int) -> dict:
        raise NotImplementedError

    def act(self, graph: GraphsTuple, params: Optional[Params] = None) -> Action:
        return self.get_qp_action(graph)[0]

    def get_qp_action(self, graph: GraphsTuple, relax_penalty: float = 1e3) -> [Action, Array]:
        assert graph.is_single  # consider single graph
        agent_node_mask = graph.node_type == 0
        agent_node_id = mask2index(agent_node_mask, self.n_agents)

        def h_aug(new_agent_state: State) -> Array:
            new_state = graph.states.at[agent_node_id].set(new_agent_state)
            new_graph = graph._replace(edges=new_state[graph.receivers] - new_state[graph.senders], states=new_state)
            val = self.get_cbf(new_graph)
            assert val.shape == (self.n_agents, self.k)
            return val

        agent_state = graph.type_states(type_idx=0, n_type=self.n_agents)
        h = h_aug(agent_state)  # (n_agents, k)
        h_x = jax.jacfwd(h_aug)(agent_state)  # (n_agents, k | n_agents, nx)
        h = h.reshape(-1)  # (n_agents * k,)

        dyn_f, dyn_g = self._env.control_affine_dyn(agent_state)
        Lf_h = ei.einsum(h_x, dyn_f, "agent_i k agent_j nx, agent_j nx -> agent_i k")
        Lg_h = ei.einsum(h_x, dyn_g, "agent_i k agent_j nx, agent_j nx nu -> agent_i k agent_j nu")
        Lf_h = Lf_h.reshape(-1)  # (n_agents * k,)
        Lg_h = Lg_h.reshape((self.n_agents * self.k, -1))  # (n_agents * k, n_agents * nu)

        u_lb, u_ub = self._env.action_lim()
        u_lb = u_lb[None, :].repeat(self.n_agents, axis=0).reshape(-1)
        u_ub = u_ub[None, :].repeat(self.n_agents, axis=0).reshape(-1)
        u_ref = self._env.u_ref(graph).reshape(-1)

        # construct QP
        H = jnp.eye(self._env.action_dim * self.n_agents + self.n_agents * self.k, dtype=jnp.float32)
        H = H.at[-self.n_agents * self.k:, -self.n_agents * self.k:].set(
            H[-self.n_agents * self.k:, -self.n_agents * self.k:] * 10.0)
        g = jnp.concatenate([-u_ref, relax_penalty * jnp.ones(self.n_agents * self.k)])
        C = -jnp.concatenate([Lg_h, jnp.eye(self.n_agents * self.k)], axis=1)
        b = Lf_h + self.alpha * h  # (n_agents * k,)

        r_lb = jnp.array([0.] * self.n_agents * self.k, dtype=jnp.float32)
        r_ub = jnp.array([jnp.inf] * self.n_agents * self.k, dtype=jnp.float32)

        l_box = jnp.concatenate([u_lb, r_lb], axis=0)
        u_box = jnp.concatenate([u_ub, r_ub], axis=0)

        qp = JaxProxQP.QPModel.create(H, g, C, b, l_box, u_box)
        settings = JaxProxQP.Settings.default()
        settings.max_iter = 100
        settings.dua_gap_thresh_abs = None
        solver = JaxProxQP(qp, settings)
        sol = solver.solve()

        assert sol.x.shape == (self.action_dim * self.n_agents + self.n_agents * self.k,)
        u_opt, r = sol.x[:self.action_dim * self.n_agents], sol.x[-self.n_agents * self.k:]
        u_opt = u_opt.reshape(self.n_agents, -1)

        return u_opt, r

    def save(self, save_dir: str, step: int):
        raise NotImplementedError

    def load(self, load_dir: str, step: int):
        raise NotImplementedError
