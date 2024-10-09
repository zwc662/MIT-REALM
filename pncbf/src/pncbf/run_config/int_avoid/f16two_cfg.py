from pncbf.dyn.f16_two import F16Two
from pncbf.ncbf.int_avoid import IntAvoidCfg, IntAvoidEvalCfg, IntAvoidTrainCfg
from pncbf.utils.schedules import Constant, Lin, SchedCtsHorizon
from pncbf.run_config.run_cfg import RunCfg
from pncbf.run_config.sac.sac_loop_cfg import SACLoopCfg


def get(seed: int) -> RunCfg[IntAvoidCfg, SACLoopCfg]:
    dt = F16Two.DT
    sched1_steps = 500_000
    sched1_warmup = 100_000
    sched1 = Lin(80, 500, sched1_steps, warmup=sched1_warmup)
    lam = SchedCtsHorizon(sched1, dt)

    lr = Constant(3e-4)
    # wd = Constant(1e-1)
    wd = Constant(1e-4)
    # act = "softplus"
    act = "tanh"

    tgt_rhs = Lin(0.0, 0.9, steps=50_000, warmup=50_000)

    collect_size = 1024
    rollout_dt = 0.1
    train_cfg = IntAvoidTrainCfg(
        collect_size,
        rollout_dt,
        rollout_T=49,
        batch_size=8192,
        lam=lam,
        tau=0.005,
        tgt_rhs=tgt_rhs,
        use_eq_state=True,
        use_grad_terms=False,
        use_hgrad=False,
    )
    eval_cfg = IntAvoidEvalCfg(eval_rollout_T=100)
    alg_cfg = IntAvoidCfg(
        act=act,
        lr=lr,
        wd=wd,
        hids=[256, 256, 256, 256],
        train_cfg=train_cfg,
        eval_cfg=eval_cfg,
        n_Vs=2,
        n_min_tgt=2,
        use_multi_norm=False,
        rescale_outputs=False,
    )
    loop_cfg = SACLoopCfg(n_iters=lam.total_steps + 50_000, ckpt_every=20_000, log_every=200, eval_every=20_000)
    return RunCfg(seed, alg_cfg, loop_cfg)


def get_pi(seed: int):
    dt = F16Two.DT
    sched1_steps = 2_000_000
    sched1_warmup = 500_000
    sched1 = Lin(80, 700, sched1_steps, warmup=sched1_warmup)
    lam = SchedCtsHorizon(sched1, dt)

    cfg = get(seed)
    cfg.alg_cfg.train_cfg.lam = lam
    cfg.alg_cfg.train_cfg.tgt_rhs = Lin(0.0, 0.9, steps=1_000_000, warmup=200_000)
    cfg.loop_cfg.n_iters = lam.total_steps + 500_000

    return cfg
