import pathlib

import jax
import jax.numpy as jnp
import scipy
from jaxproxqp.jaxproxqp import JaxProxQP

from pncbf.dyn.dyn_types import Control, State
from pncbf.dyn.task import Task
from pncbf.qp.min_norm_cbf import min_norm_cbf
from pncbf.utils.jax_types import FloatScalar


class MatlabSOS:
    def __init__(self, task: Task, mat_path: pathlib.Path, coef: float = 1e-3):
        self.mat = scipy.io.loadmat(str(mat_path))
        self.h_coefs = self.mat["h_coefs_double"]
        self.coef = coef
        self.task = task

    def __call__(self, x: State) -> FloatScalar:
        return self.get_B(x)

    def get_B(self, x: State) -> FloatScalar:
        assert x.shape == (4,)
        x1, x2, x3, x4 = x

        if self.h_coefs.shape == (3, 3, 3, 3):
            x1_terms = jnp.array([x1**2, x1, 1])
            x2_terms = jnp.array([x2**2, x2, 1])
            x3_terms = jnp.array([x3**2, x3, 1])
            x4_terms = jnp.array([x4**2, x4, 1])
        else:
            assert self.h_coefs.shape == (7, 7, 7, 7)
            x1_terms = jnp.array([x1**6, x1**5, x1**4, x1**3, x1**2, x1, 1])
            x2_terms = jnp.array([x2**6, x2**5, x2**4, x2**3, x2**2, x2, 1])
            x3_terms = jnp.array([x3**6, x3**5, x3**4, x3**3, x3**2, x3, 1])
            x4_terms = jnp.array([x4**6, x4**5, x4**4, x4**3, x4**2, x4, 1])

        # (3, 3, 3, 3)
        x1_terms = x1_terms[:, None, None, None]
        x2_terms = x2_terms[None, :, None, None]
        x3_terms = x3_terms[None, None, :, None]
        x4_terms = x4_terms[None, None, None, :]
        h_terms = x1_terms * x2_terms * x3_terms * x4_terms
        assert h_terms.shape == self.h_coefs.shape
        return -self.coef * jnp.sum(h_terms * self.h_coefs)

    def cbf(self, state: State, nom_pol, alpha: float) -> Control:
        h_B = self.get_B(state)
        hx_Bx = jax.jacfwd(self.get_B)(state)

        # Compute QP sol.
        u_nom = nom_pol(state)
        u_lb, u_ub = self.task.u_min, self.task.u_max
        f, G = self.task.f(state), self.task.G(state)

        settings = JaxProxQP.Settings.default()
        # settings.max_iter = 15
        # settings.max_iter_in = 5
        settings.max_iterative_refine = 5
        u_qp, r, sol = min_norm_cbf(alpha, u_lb, u_ub, h_B, hx_Bx, f, G, u_nom, settings=settings)
        u_qp = self.task.chk_u(u_qp.clip(u_lb, u_ub))
        return u_qp
