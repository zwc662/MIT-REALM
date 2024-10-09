from clfrl.ncbf.avoid_fixed import AvoidFixedCfg, AvoidFixedEvalCfg, AvoidFixedTrainCfg
from clfrl.utils.schedules import Constant, ExpDecay, Lin, LinDecay, SchedCtsHorizon, SchedEffHorizon
from clfrl.run_config.run_cfg import RunCfg
from clfrl.run_config.sac.sac_loop_cfg import SACLoopCfg


def get(seed: int) -> RunCfg[AvoidFixedCfg, SACLoopCfg]:
    # lam = ExpDecay(1.0, 5_000, 0.7, 25_000, staircase=False, end_value=1e-4)
    lam = SchedCtsHorizon(Lin(40, 150, 450_000, warmup=50_000), 0.1)
    train_cfg = AvoidFixedTrainCfg(batch_size=8192, lam=lam, compl_max=True)
    # Segway has a super long "steps to collision", so we need a much longer eval rollout length.
    # eval_cfg = AvoidFixedEvalCfg(eval_rollout_T=500)
    eval_cfg = AvoidFixedEvalCfg(eval_rollout_T=60)
    alg_cfg = AvoidFixedCfg(act="tanh", lr=Constant(3e-4), hids=[256, 256], train_cfg=train_cfg, eval_cfg=eval_cfg)
    loop_cfg = SACLoopCfg(n_iters=500_000, ckpt_every=5_000, log_every=100, eval_every=1_000, plot_every=5_000)
    return RunCfg(seed, alg_cfg, loop_cfg)
