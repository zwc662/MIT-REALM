import jax.numpy as jnp

from pncbf.utils.jax_types import FloatScalar, Vec3


def sdf_capped_cone(pt: Vec3, a: Vec3, b: Vec3, ra: FloatScalar, rb: FloatScalar) -> FloatScalar:
    eps = 1e-5

    rba = rb - ra
    baba = jnp.dot(b - a, b - a)
    papa = jnp.dot(pt - a, pt - a)
    paba = jnp.dot(pt - a, b - a) / (baba + eps)

    x = jnp.sqrt(papa - paba * paba * baba)

    cax = jnp.maximum(0.0, x - jnp.where(paba < 0.5, ra, rb))
    cay = jnp.abs(paba - 0.5) - 0.5

    # Should be nonzero.
    k = rba * rba + baba
    f = jnp.clip((rba * (x - ra) + paba * baba) / (k + eps), 0.0, 1.0)

    cbx = x - ra - f * rba
    cby = paba - f
    s = jnp.where((cbx < 0.0) & (cay < 0.0), -1.0, 1.0)
    return s * jnp.sqrt(jnp.minimum(cax * cax + cay * cay * baba, cbx * cbx + cby * cby * baba))
