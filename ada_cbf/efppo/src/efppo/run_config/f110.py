from efppo.rl.collector import CollectorCfg
from efppo.rl.efppo_inner import EFPPOCfg
from efppo.rl.baseline import BaselineCfg, BaselineSAC, BaselineDQN
from efppo.utils.schedules import LinDecay

from enum import Enum

# Enum mapping strings to classes
class BaselineEnum(Enum):
    SAC = BaselineSAC
    DQN = BaselineDQN



def get(alg: str = 'efppo'):
    zmin, zmax = -1.0, 2.5
    nz_enc = 8
    z_mean = 0.5
    z_scale = 1.0


    pol_hids = val_hids = [256, 256, 256]

    pol_lr = LinDecay(8e-4, 8.0, warmup_steps=500_000, trans_steps=2_000_000)
    val_lr = LinDecay(8e-4, 8.0, warmup_steps=500_000, trans_steps=2_000_000)
    temp_lr = 3e-4
    entropy_cf = LinDecay(1e-2, 5e2, warmup_steps=200_000, trans_steps=1_000_000)
    disc_gamma = 0.98

    
    n_batches = 8

    batch_size = 256
    

    
    net_cfg = EFPPOCfg.NetCfg(
        pol_lr, val_lr, entropy_cf, disc_gamma, "tanh", pol_hids, val_hids, nz_enc, z_mean, z_scale
    )
    
    train_cfg = EFPPOCfg.TrainCfg(zmin, zmax, 0.95, 50.0, n_batches, 0.1, 1.0, 1.0)
    eval_cfg = EFPPOCfg.EvalCfg()
    alg_cfg = EFPPOCfg(net_cfg, train_cfg, eval_cfg)

    
    if 'baseline' in alg:
        if 'sac' in alg:
            alg = BaselineEnum.SAC
        elif 'dqn' in alg:
            alg = BaselineEnum.DQN
        n_critics = 30
        bc_ratio = 0.
        train_cfg = BaselineCfg.TrainCfg(zmin, zmax, n_batches, batch_size, bc_ratio, 1.0, 1.0)
        net_cfg = BaselineCfg.NetCfg(pol_lr, val_lr, entropy_cf, disc_gamma, "tanh", pol_hids, val_hids, nz_enc, z_mean, z_scale, n_critics = n_critics)
        alg_cfg = BaselineCfg(alg, net_cfg, train_cfg, eval_cfg)
        
    n_envs = 1
    rollout_T = 128
    mean_age = 1024
    max_T = 256
    collect_cfg = CollectorCfg(n_envs, rollout_T, mean_age, max_T)

    return alg_cfg, collect_cfg
