from typing import Generic, Protocol, TypeVar

import jax.numpy as jnp
from attrs import define

from clfrl.utils.jax_types import BFloat, FloatScalar

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
