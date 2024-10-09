from typing import Literal

from clfrl.utils.jax_types import AnyFloat

QhReductionType = Literal["min", "max", "mean"]


def apply_Qh_reduction(Qh: AnyFloat, method: QhReductionType, axis: int) -> AnyFloat:
    if method == "min":
        return Qh.min(axis=axis)
    if method == "max":
        return Qh.max(axis=axis)
    if method == "mean":
        return Qh.mean(axis=axis)

    raise NotImplementedError("{}".format(method))
