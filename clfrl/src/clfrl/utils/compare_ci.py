from typing import Any, NamedTuple

from clfrl.dyn.dyn_types import BBTControl, BBTHFloat, BBTState
from clfrl.utils.jax_types import BBFloat


class CIData(NamedTuple):
    name: str
    task_name: str
    setup_idx: int

    bbT_x: BBTState
    bbT_u: BBTControl
    bbTh_h: BBTHFloat

    bb_Xs: BBFloat
    bb_Ys: BBFloat

    notes: dict[str, Any] = {}
