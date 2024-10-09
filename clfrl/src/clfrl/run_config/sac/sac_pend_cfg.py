from clfrl.rl.collector import CollectorCfg
from clfrl.rl.sac import SACCfg, SACEvalCfg, SACTrainCfg
from clfrl.utils.schedules import Constant
from clfrl.run_config.run_cfg import RunCfg
from clfrl.run_config.sac.sac_loop_cfg import SACLoopCfg


def get(seed: int) -> RunCfg[SACCfg, SACLoopCfg]:
    collect_cfg = CollectorCfg(collect_batch=256, collect_len=16, max_age=32, mean_age=17)
    train_cfg = SACTrainCfg(
        disc_gamma=0.99,
        tau=0.005,
        batch_size=4096,
        val_pol_ratio=4,
        val_act_ratio=8,
        target_entropy=None,
    )
    eval_cfg = SACEvalCfg(eval_rollout_T=50)
    alg_cfg = SACCfg(
        n_qs=8,
        n_min_qs=2,
        act="tanh",
        pol_lr=Constant(3e-4),
        val_lr=Constant(3e-4),
        temp_lr=Constant(3e-4),
        pol_hids=[256, 256],
        val_hids=[256, 256],
        init_temp=1.0,
        critic_layernorm=True,
        collect_cfg=collect_cfg,
        train_cfg=train_cfg,
        eval_cfg=eval_cfg,
    )
    loop_cfg = SACLoopCfg(n_iters=100_000, ckpt_every=5_000)
    return RunCfg(seed, alg_cfg, loop_cfg)
