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
    # 3s is probably a long enough horizon.
    dt = 0.1
    sched1_steps = 70_000
    sched1_warmup = 40_000
    sched1 = Lin(20, 100, sched1_steps, warmup=sched1_warmup)
    lam = SchedCtsHorizon(sched1, dt)

    final_lam = horizon_to_lam(100, dt=dt)
    sched2_steps = 100_000
    lam = JoinSched(lam, Lin(final_lam, 0.0, sched2_steps), sched1.total_steps)

    # lam = SchedCtsHorizon(ExpDecay(10, 200, 75_000, warmup=25_000), 0.1)
    # lam = Constant(0.1)
    # lam = Constant(0.01)
    # train_cfg = IntAvoidTrainCfg(collect_size=8192, rollout_T=63, batch_size=8192, lam=lam, tau=0.005)

    lr = Constant(3e-4)
    wd = Constant(1e-1)

    collect_size = 16_384
    rollout_dt = 0.1
    train_cfg = IntAvoidTrainCfg(collect_size, rollout_dt, rollout_T=49, batch_size=8192, lam=lam, tau=0.005)
    eval_cfg = IntAvoidEvalCfg(eval_rollout_T=100)
    alg_cfg = IntAvoidCfg(
        act="tanh", lr=lr, wd=wd, hids=[256, 256, 256], train_cfg=train_cfg, eval_cfg=eval_cfg, n_Vs=2, n_min_tgt=2
    )
    loop_cfg = SACLoopCfg(n_iters=lam.total_steps + 50_000, ckpt_every=5_000, log_every=100, eval_every=5_000)
    return RunCfg(seed, alg_cfg, loop_cfg)
