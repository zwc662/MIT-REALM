from clfrl.ncbf_pol.ncbf_pol import NCBFPolCfg, NCBFPolEvalCfg, NCBFPolTrainCfg
from clfrl.utils.schedules import Constant
from clfrl.run_config.run_cfg import RunCfg
from clfrl.run_config.sac.sac_loop_cfg import SACLoopCfg


def get(seed: int) -> RunCfg[NCBFPolCfg, SACLoopCfg]:
    lam = 1.0
    train_cfg = NCBFPolTrainCfg(
        batch_size=8192, lam=lam, unsafe_eps=0.1, safe_eps=0.1, descent_eps=0.1, use_eq_state=True
    )
    eval_cfg = NCBFPolEvalCfg(eval_rollout_T=64)
    alg_cfg = NCBFPolCfg(act="tanh", lr=Constant(3e-4), hids=[256, 256], train_cfg=train_cfg, eval_cfg=eval_cfg)
    loop_cfg = SACLoopCfg(n_iters=50_000, ckpt_every=5_000, log_every=100, eval_every=5_000)
    return RunCfg(seed, alg_cfg, loop_cfg)
