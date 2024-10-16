import functools as ft
from typing import NamedTuple

import jax.lax as lax
import jax.numpy as jnp
import numpy as np

from pncbf.dyn.dyn_types import HFloat, THFloat
from pncbf.utils.jax_types import FloatScalar, TFloat
from pncbf.utils.jax_utils import jax_vmap

   

class AllDiscAvoidTerms(NamedTuple):
    """Terms needed to compute V(x_t) = max{ h_max_lhs, h_disc_int_rhs + discount * V(x_T) } for all t."""

    Th_disc_tot: THFloat # Total discounted cost until h > 0
    Th_is_hgt0: THFloat # Is h > 0 mask 
    Th_Vh_tgt: THFloat # Targeted value

 
 
def compute_all_disc_avoid_terms(lambd: FloatScalar, dt: FloatScalar, Th_h: THFloat) -> AllDiscAvoidTerms:
    """
    Compute the targeted value function for all states in a trajectory
    Since the dynamics is deterministic, we can compute the value function with the discounted accumulated cost
    The state action cost is -1 if h_t <= 0, otherwise 0
    For each state in a trajectory, its future costs are accumulated until h_t is greater than 0
    The targeted value function for x_t is its future accumulated cost if h(x_t) >= 0, otherwise 0
    """
    T, nh = Th_h.shape
    gamma = jnp.exp(-lambd * dt)

    # Compute the mask where h > 0
    Th_is_hgt0 = Th_h > 0

    # Mark the state for which h_t > 0
    Th_is_hgt0 = Th_h > 0

    # Compute the length of segments to the first h_t > 0 in the future
    Th_segment_length = jnp.cumsum(Th_is_hgt0[::-1], axis=0)[::-1]

    # Define a function to compute the accumulated cost for each state
    def compute_accumulated_cost(segment_length):
        indices = jnp.arange(segment_length)
        return -jnp.sum(gamma ** indices)
    # Vectorize the computation of accumulated cost
    compute_accumulated_cost_vmap = jax_vmap(compute_accumulated_cost)
      
    # Initialize the carry and scan over the trajectory
    init_carry = (jnp.zeros(nh), Th_h[-1])
    (_, _), Th_disc_tot = lax.scan(scan_fn, init_carry, Th_h[:-1], reverse=True)

    # Append the last value to match the original shape
    Th_disc_tot = jnp.concatenate([Th_disc_tot, jnp.zeros((1, nh))], axis=0)

    # Compute the targeted value
    Th_Vh_tgt = Th_disc_tot * Th_is_hgt0

    return AllDiscAvoidTerms(Th_disc_tot=Th_disc_tot, Th_is_hgt0=Th_is_hgt0, Th_Vh_tgt=Th_Vh_tgt)