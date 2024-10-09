from clfrl.ncbf.ncbf import NCBFCfg, NCBFEvalCfg, NCBFTrainCfg
from clfrl.utils.schedules import Constant
from clfrl.run_config.run_cfg import RunCfg
from clfrl.run_config.sac.sac_loop_cfg import SACLoopCfg


def get(seed: int) -> RunCfg[NCBFCfg, SACLoopCfg]:
    lam = 1.0

    use_eq_state = "eq_state"
    # use_eq_state = "none"

    train_cfg = NCBFTrainCfg(batch_size=8192, lam=lam, unsafe_eps=0.1, safe_eps=0.1, use_eq_state=use_eq_state)
    eval_cfg = NCBFEvalCfg(eval_rollout_T=64)
    alg_cfg = NCBFCfg(act="tanh", lr=Constant(3e-4), hids=[256, 256], train_cfg=train_cfg, eval_cfg=eval_cfg)
    loop_cfg = SACLoopCfg(n_iters=50_000, ckpt_every=5_000, log_every=100, eval_every=5_000)
    return RunCfg(seed, alg_cfg, loop_cfg)
