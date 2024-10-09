import einops as ei
import jax.numpy as jnp
import jaxopt
import numpy as np

from clfrl.dyn.dyn_types import Control, State
from clfrl.solvers.qp import jaxopt_osqp
from clfrl.utils.jax_types import FloatScalar


def min_norm_control(
    desc_lambda: float,
    u_lb: Control,
    u_ub: Control,
    V: FloatScalar,
    Vx: State,
    f: State,
    G,
    penalty: float = 10.0,
    relax_eps1: float = 1e-2,
    relax_eps2: float = 5e-1,
    maxiter: int = 100,
    tol: float = 5e-4,
):
    nx, nu = G.shape
    Lf_V = jnp.dot(Vx, f)
    LG_V = ei.einsum(Vx, G, "nx, nx nu -> nu")

    Q = np.eye(nu + 1)
    # 0.5 * penalty * ( r + relax_eps )^2
    Q[-1, -1] = penalty
    c = np.zeros(nu + 1)
    c[-1] = penalty * relax_eps2

    # Relaxed descent condition. LfV + LGV u + lambda V - r <= 0
    G1 = jnp.concatenate([LG_V, -jnp.ones(1)], axis=0)
    h1 = -(Lf_V + desc_lambda * V)

    # r >= -eps1     <=>      -r <= eps1
    G2 = np.zeros(nu + 1)
    G2[-1] = -1
    h2 = relax_eps1

    G = jnp.stack([G1, G2], axis=0)
    h = jnp.array([h1, h2])

    solver = jaxopt.OSQP(eq_qp_solve="lu", maxiter=maxiter, tol=tol)

    params, state, info = jaxopt_osqp(solver, Q, c, G, h, u_lb, u_ub)
    u_opt, r = params.primal[:nu], params.primal[-1]
    assert u_opt.shape == (nu,)
    return u_opt, r, info


def bangbang_control(u_lb: Control, u_ub: Control, Vx: State, G):
    LG_V = ei.einsum(Vx, G, "nx, nx nu -> nu")
    # To minimize, if LG_V is negative, we want u to be as large as possible.
    u_opt = jnp.where(LG_V < 0, u_ub, u_lb)
    return u_opt


def min_norm_control_interp(
    desc_lambda: float,
    interp_frac: float,
    u_lb: Control,
    u_ub: Control,
    V: FloatScalar,
    Vx: State,
    f: State,
    G,
    penalty: float = 10.0,
    relax_eps1: float = 1e-2,
    relax_eps2: float = 5e-1,
    maxiter: int = 100,
    tol: float = 5e-4,
    *args,
    **kwargs
):
    """Interpolates between min_norm (0) and bang-bang (1) in Vdot space."""
    nx, nu = G.shape
    LG_V = ei.einsum(Vx, G, "nx, nx nu -> nu")

    u_minnorm, r_minnorm, info_minnorm = min_norm_control(
        desc_lambda, u_lb, u_ub, V, Vx, f, G, penalty, relax_eps1, relax_eps2, maxiter, tol
    )
    u_bangbang = bangbang_control(u_lb, u_ub, Vx, G)

    if interp_frac <= 0:
        return u_minnorm, r_minnorm, info_minnorm
    if interp_frac >= 1:
        return u_bangbang, None, None

    LG_V_minnorm = jnp.dot(LG_V, u_minnorm)
    LG_V_bangbang = jnp.dot(LG_V, u_bangbang)

    LG_V_desired = (1 - interp_frac) * LG_V_minnorm + interp_frac * LG_V_bangbang

    # Minnorm, but with constraint on LG_V instead of descent condition.
    Q = np.eye(nu)
    c = np.zeros(nu)

    G = LG_V[None, :]
    h = jnp.array([LG_V_desired])

    solver = jaxopt.OSQP(eq_qp_solve="lu", maxiter=maxiter, tol=tol)
    params, state, info = jaxopt_osqp(solver, Q, c, G, h, u_lb, u_ub)
    u_opt, r = params.primal[:nu], params.primal[-1]
    assert u_opt.shape == (nu,)
    return u_opt, r, info


def min_norm_control_fixed(
    desc_rate: float,
    u_lb: Control,
    u_ub: Control,
    Vx: State,
    f: State,
    G,
    penalty: float = 10.0,
    relax_eps1: float = 1e-2,
    relax_eps2: float = 5e-1,
    maxiter: int = 100,
    tol: float = 5e-4,
):
    nx, nu = G.shape
    Lf_V = jnp.dot(Vx, f)
    LG_V = ei.einsum(Vx, G, "nx, nx nu -> nu")

    Q = np.eye(nu + 1)
    Q[-1, -1] = penalty
    c = np.zeros(nu + 1)
    c[-1] = penalty * relax_eps2

    # Relaxed descent condition. LfV + LGV u + desc_rate - r <= 0
    G1 = jnp.concatenate([LG_V, -jnp.ones(1)], axis=0)
    h1 = -(Lf_V + desc_rate)

    # r >= -eps1     <=>      -r <= eps1
    G2 = np.zeros(nu + 1)
    G2[-1] = -1
    h2 = relax_eps1

    G = jnp.stack([G1, G2], axis=0)
    h = jnp.array([h1, h2])

    solver = jaxopt.OSQP(eq_qp_solve="lu", maxiter=maxiter, tol=tol)

    params, state, info = jaxopt_osqp(solver, Q, c, G, h, u_lb, u_ub)
    u_opt, r = params.primal[:nu], params.primal[-1]
    assert u_opt.shape == (nu,)
    return u_opt, r, info


def min_norm_control_minrate(
    desc_rate: float,
    desc_lambd: float,
    u_lb: Control,
    u_ub: Control,
    V: FloatScalar,
    Vx: State,
    f: State,
    G,
    penalty: float = 10.0,
    relax_eps1: float = 1e-2,
    relax_eps2: float = 5e-1,
    maxiter: int = 100,
    tol: float = 5e-4,
):
    nx, nu = G.shape
    Lf_V = jnp.dot(Vx, f)
    LG_V = ei.einsum(Vx, G, "nx, nx nu -> nu")

    Q = np.eye(nu + 1)
    Q[-1, -1] = penalty
    c = np.zeros(nu + 1)
    c[-1] = penalty * relax_eps2

    # Relaxed descent condition. LfV + LGV u + min(desc_rate, lambd * V) - r <= 0
    G1 = jnp.concatenate([LG_V, -jnp.ones(1)], axis=0)
    h1 = -(Lf_V + jnp.minimum(desc_rate, desc_lambd * V))

    # r >= -eps1     <=>      -r <= eps1
    G2 = np.zeros(nu + 1)
    G2[-1] = -1
    h2 = relax_eps1

    G = jnp.stack([G1, G2], axis=0)
    h = jnp.array([h1, h2])

    solver = jaxopt.OSQP(eq_qp_solve="lu", maxiter=maxiter, tol=tol)

    params, state, info = jaxopt_osqp(solver, Q, c, G, h, u_lb, u_ub)
    u_opt, r = params.primal[:nu], params.primal[-1]
    assert u_opt.shape == (nu,)
    return u_opt, r, info
