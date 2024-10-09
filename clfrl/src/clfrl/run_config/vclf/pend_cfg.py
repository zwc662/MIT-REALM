from clfrl.nclf.vclf import VCLFCfg, VCLFEvalCfg, VCLFTrainCfg
from clfrl.utils.schedules import Constant
from clfrl.run_config.run_cfg import RunCfg
from clfrl.run_config.sac.sac_loop_cfg import SACLoopCfg


def get(seed: int) -> RunCfg[VCLFCfg, SACLoopCfg]:
    train_cfg = VCLFTrainCfg(batch_size=8192, desc_rate=0.05, goal_zero_margin=1e-3, stop_u=False)
    eval_cfg = VCLFEvalCfg(eval_rollout_T=64)
    alg_cfg = VCLFCfg(act="tanh", lr=Constant(3e-4), hids=[256, 256], train_cfg=train_cfg, eval_cfg=eval_cfg)
    loop_cfg = SACLoopCfg(n_iters=200_000, ckpt_every=5_000, log_every=500)
    return RunCfg(seed, alg_cfg, loop_cfg)
