from typing import Any, Generic, TypeVar

from attrs import asdict, define
from typing_extensions import Self

from run_config.run_cfg import LoopCfg
 

@define
class SACLoopCfg(Generic[LoopCfg]):
    n_iters: int
    ckpt_every: int
    log_every: int
    eval_every: int 

    def asdict(self) -> dict[str, Any]:
        return asdict(self)
