from typing import Generic, Protocol, TypeVar

import jax.numpy as jnp
from attrs import define

from pncbf.utils.jax_types import BFloat, FloatScalar, Vec3

ConstrDomain = TypeVar("ConstrDomain", bound=FloatScalar)


class Constraint(Protocol[ConstrDomain]):
    def to_array(self, x: ConstrDomain) -> BFloat:
        """Get the h value. Negative is safe."""
        ...


@define
class BoxConstraint(Constraint[FloatScalar]):
    lb: FloatScalar
    ub: FloatScalar

    def to_array(self, x: FloatScalar) -> BFloat:
        return jnp.array([self.lb - x, x - self.ub])


@define
class LowerBound(Constraint[FloatScalar]):
    lb: FloatScalar

    def to_array(self, x: FloatScalar) -> BFloat:
        return jnp.array([self.lb - x])


@define
class UpperBound(Constraint[FloatScalar]):
    ub: FloatScalar

    def to_array(self, x: FloatScalar) -> BFloat:
        return jnp.array([x - self.ub])


@define
class PositiveConstraint(Constraint[FloatScalar]):
    def to_array(self, x: FloatScalar) -> BFloat:
        assert x.shape == tuple()
        return -jnp.atleast_1d(x)


def to_vec_mag_exp(vec: Vec3, magnitude_scale: FloatScalar):
    """Convert a vector to the unit vector and the exponential transform of the magnitude.
    vec: (3, )
    magnitude_scala: 0 - 0.99 covers roughly 4.5 times this scale.
    """
    assert vec.ndim == 1
    norm = jnp.linalg.norm(vec)
    # Handle the division by 0 case.
    norm = norm.clip(min=1e-6)
    unit_vec = vec / norm
    mag_obs = jnp.exp(-norm / magnitude_scale)
    return jnp.concatenate([unit_vec, jnp.atleast_1d(mag_obs)])
