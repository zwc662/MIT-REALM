from clfrl.dyn.doubleint_wall import DoubleIntWall
from clfrl.ncbf.int_avoid import IntAvoidCfg, IntAvoidEvalCfg, IntAvoidTrainCfg
from clfrl.utils.schedules import (
    Constant,
    ExpDecay,
    JoinSched,
    Lin,
    LinDecay,
    SchedCtsHorizon,
    SchedEffHorizon,
    horizon_to_lam,
)
from clfrl.run_config.run_cfg import RunCfg
from clfrl.run_config.sac.sac_loop_cfg import SACLoopCfg


def get(seed: int) -> RunCfg[IntAvoidCfg, SACLoopCfg]:
    # lam = ExpDecay(1.0, 5_000, 0.7, 25_000, staircase=False, end_value=1e-4)
    dt = DoubleIntWall.DT

    sched1_steps = 25_000
    sched1_warmup = 15_000

    horizon_end = 200
    sched1 = Lin(10, 50, sched1_steps, warmup=sched1_warmup)
    lam = SchedCtsHorizon(sched1, dt)
    final_lam = horizon_to_lam(horizon_end, dt=dt)

    sched2_steps = 10_000
    lam = JoinSched(lam, Lin(final_lam, 0.0, sched2_steps), sched1.total_steps)
    # lam = Constant(0.3)

    lr = Constant(3e-4)
    wd = Constant(1e-1)

    act = "tanh"
    # act = "gelu"

    # lam = SchedCtsHorizon(Lin(10, 200, 75_000, warmup=25_000), 0.1)
    # lam = SchedCtsHorizon(ExpDecay(10, 200, 75_000, warmup=25_000), 0.1)
    # lam = Constant(0.1)
    # lam = Constant(0.01)
    collect_size = 8192
    rollout_dt = dt
    train_cfg = IntAvoidTrainCfg(collect_size, rollout_dt, rollout_T=24, batch_size=8192, lam=lam, tau=0.005)
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
    )
    loop_cfg = SACLoopCfg(n_iters=50_000, ckpt_every=5_000, log_every=100, eval_every=5_000)
    return RunCfg(seed, alg_cfg, loop_cfg)
