from clfrl.nclf.nclf_fixed import NCLFFixedCfg, NCLFFixedEvalCfg, NCLFFixedTrainCfg
from clfrl.utils.schedules import Constant
from clfrl.run_config.run_cfg import RunCfg
from clfrl.run_config.sac.sac_loop_cfg import SACLoopCfg


def get(seed: int) -> RunCfg[NCLFFixedCfg, SACLoopCfg]:
    train_cfg = NCLFFixedTrainCfg(batch_size=8192, desc_lam=0.1, desc_rate=1.0, goal_zero_margin=1e-3, use_rate=True)
    eval_cfg = NCLFFixedEvalCfg(eval_rollout_T=64)
    alg_cfg = NCLFFixedCfg(act="tanh", lr=Constant(3e-4), hids=[256, 256], train_cfg=train_cfg, eval_cfg=eval_cfg)
    loop_cfg = SACLoopCfg(n_iters=200_000, ckpt_every=5_000, log_every=100)
    return RunCfg(seed, alg_cfg, loop_cfg)
