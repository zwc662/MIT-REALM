from pncbf.dyn.doubleint_wall import DoubleIntWall
from pncbf.ncbf.int_avoid import IntAvoidCfg, IntAvoidEvalCfg, IntAvoidTrainCfg
from pncbf.utils.schedules import (
    Constant,
    ExpDecay,
    JoinSched,
    Lin,
    LinDecay,
    SchedCtsHorizon,
    SchedEffHorizon,
    horizon_to_lam,
)
from pncbf.run_config.run_cfg import RunCfg
from pncbf.run_config.sac.sac_loop_cfg import SACLoopCfg


def get(seed: int) -> RunCfg[IntAvoidCfg, SACLoopCfg]:
    # lam = ExpDecay(1.0, 5_000, 0.7, 25_000, staircase=False, end_value=1e-4)
    dt = DoubleIntWall.DT

    sched1_steps = 35_000
    sched1_warmup = 15_000

    # horizon_end = 200
    sched1 = Lin(10, 25, sched1_steps, warmup=sched1_warmup)
    lam = SchedCtsHorizon(sched1, dt)

    lr = Constant(3e-4)
    wd = Constant(1e-1)

    # act = "tanh"
    act = "softplus"
    # act = "gelu"

    # tgt_rhs = Constant(0.9)
    tgt_rhs = Lin(0.0, 0.9, steps=35_000, warmup=15_000)

    collect_size = 8192
    rollout_dt = dt
    train_cfg = IntAvoidTrainCfg(
        collect_size, rollout_dt, rollout_T=24, batch_size=8192, lam=lam, tau=0.005, tgt_rhs=tgt_rhs, use_eq_state=False
    )
    eval_cfg = IntAvoidEvalCfg(eval_rollout_T=64)
    alg_cfg = IntAvoidCfg(
        act=act,
        lr=lr,
        wd=wd,
        hids=[256, 256],
        train_cfg=train_cfg,
        eval_cfg=eval_cfg,
        n_Vs=2,
        n_min_tgt=2,
        use_multi_norm=False,
        rescale_outputs=False,
    )
    loop_cfg = SACLoopCfg(n_iters=lam.total_steps + 5_000, ckpt_every=5_000, log_every=100, eval_every=5_000)
    return RunCfg(seed, alg_cfg, loop_cfg)


def get_pi(seed: int) -> RunCfg[IntAvoidCfg, SACLoopCfg]:
    dt = DoubleIntWall.DT

    sched1_steps = 30_000
    sched1_warmup = 5_000
    sched1 = Lin(10, 100, sched1_steps, warmup=sched1_warmup)
    lam = SchedCtsHorizon(sched1, dt)
    # final_lam = horizon_to_lam(50, dt=dt)

    # sched2_steps = 5_000
    # lam = JoinSched(lam, Lin(final_lam, 0.0, sched2_steps), sched1.total_steps)

    tgt_rhs = Lin(0.0, 0.99, steps=10_000, warmup=5_000)

    cfg = get(seed)
    cfg.alg_cfg.train_cfg.lam = lam
    cfg.alg_cfg.train_cfg.tgt_rhs = tgt_rhs
    cfg.loop_cfg.n_iters = lam.total_steps + 5_000

    return cfg
