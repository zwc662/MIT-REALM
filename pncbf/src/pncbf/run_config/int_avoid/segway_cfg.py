from pncbf.dyn.segway import Segway
from pncbf.ncbf.int_avoid import IntAvoidCfg, IntAvoidEvalCfg, IntAvoidTrainCfg
from pncbf.utils.schedules import (
    Constant,
    JoinSched,
    Lin,
    SchedCtsHorizon,
    horizon_to_lam,
)
from pncbf.run_config.run_cfg import RunCfg
from pncbf.run_config.sac.sac_loop_cfg import SACLoopCfg


def get(seed: int) -> RunCfg[IntAvoidCfg, SACLoopCfg]:
    # lam = ExpDecay(0.15, 10_000, 0.7, 25_000, staircase=False, end_value=1e-3)
    dt = Segway.DT
    sched1_steps = 50_000
    sched1_warmup = 20_000
    sched1 = Lin(60, 150, sched1_steps, warmup=sched1_warmup)
    lam = SchedCtsHorizon(sched1, dt)

    # final_lam = horizon_to_lam(200, dt=0.1)
    # sched2_steps = 50_000
    # lam = JoinSched(lam, Lin(final_lam, 0.0, sched2_steps), sched1.total_steps)

    lr = Constant(3e-4)
    wd = Constant(1e-1)

    tgt_rhs = Lin(0.0, 0.9, steps=50_000, warmup=20_000)

    collect_size = 16_384
    rollout_dt = Segway.DT
    train_cfg = IntAvoidTrainCfg(
        collect_size, rollout_dt, rollout_T=99, batch_size=8192, lam=lam, tau=0.005, tgt_rhs=tgt_rhs, use_eq_state=False
    )
    eval_cfg = IntAvoidEvalCfg(eval_rollout_T=64)
    alg_cfg = IntAvoidCfg(
        act="softplus",
        lr=lr,
        wd=wd,
        hids=[256, 256, 256],
        train_cfg=train_cfg,
        eval_cfg=eval_cfg,
        n_Vs=2,
        n_min_tgt=2,
        use_multi_norm=False,
        rescale_outputs=False,
    )
    loop_cfg = SACLoopCfg(n_iters=lam.total_steps + 10_000, ckpt_every=5_000, log_every=100, eval_every=2_500)
    return RunCfg(seed, alg_cfg, loop_cfg)


def get_pi(seed: int) -> RunCfg[IntAvoidCfg, SACLoopCfg]:
    dt = Segway.DT

    sched1_steps = 50_000
    sched1_warmup = 20_000
    sched1 = Lin(60, 150, sched1_steps, warmup=sched1_warmup)
    lam = SchedCtsHorizon(sched1, dt)
    final_lam = horizon_to_lam(150, dt=dt)

    sched2_steps = 50_000
    lam = JoinSched(lam, Lin(final_lam, 0.0, sched2_steps), sched1.total_steps)

    tgt_rhs = Lin(0.0, 0.9, steps=50_000, warmup=20_000)

    cfg = get(seed)
    cfg.alg_cfg.train_cfg.lam = lam
    cfg.alg_cfg.train_cfg.tgt_rhs = tgt_rhs
    cfg.loop_cfg.n_iters = lam.total_steps + 5_000

    return cfg
