from attrs import define


@define
class SACLoopCfg:
    n_iters: int
    eval_every: int = 1_000
    log_every: int = 50
    plot_every: int = 1_000
    ckpt_every: int = 5_000
    start_train: int = 500
    rb_capacity: int = 5_000_000
