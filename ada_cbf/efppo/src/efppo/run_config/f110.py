from efppo.rl.collector import CollectorCfg
from efppo.rl.efppo_inner import EFPPOCfg
from efppo.rl.baseline import BaselineCfg, BaselineSAC, BaselineSACDisc, BaselineDQN
from efppo.utils.schedules import LinDecay

from enum import Enum



def get(name: str = 'efppo'):
    zmin, zmax = 2.4, 2.5 #-1.0, 2.5
    nz_enc = 8
    z_mean = 2.5 #0.5
    z_scale = 0.1 #1.0


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

    alg = None
    if 'baseline' in name:
        if 'sac' in name:
            if 'disc' in name:
                alg = BaselineSACDisc
            else:
                alg = BaselineSAC
        elif 'dqn' in name:
            alg = BaselineDQN
        
        n_critics = 2
        bc_ratio = 0.
         

        train_cfg = BaselineCfg.TrainCfg(zmin, zmax, n_batches, batch_size, bc_ratio, 1.0, 1.0)
        net_cfg = BaselineCfg.NetCfg(pol_lr, val_lr, entropy_cf, disc_gamma, "tanh", pol_hids, val_hids, nz_enc, z_mean, z_scale, n_critics = n_critics)
        alg_cfg = BaselineCfg(alg, net_cfg, train_cfg, eval_cfg)
    assert alg is not None
    
    n_envs = 1
    rollout_T = 128
    mean_age = 1024
    max_T = 2000
    collect_cfg = CollectorCfg(n_envs, rollout_T, mean_age, max_T)

    return alg_cfg, collect_cfg
