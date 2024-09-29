import ipdb
import jax.random as jr
import typer
from typing_extensions import Annotated, Optional

import sys
import os

import re

import argparse

import git
repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha

import pickle 

from datetime import datetime

# Get current timestamp in yyy_mm_dd format
current_timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


import efppo.run_config.f110
from efppo.rl.baseline import Baseline, BaselineSAC, BaselineSACDisc, BaselineDQN, BaselineDQNCBF
from efppo.rl.efppo_inner_trainer import EFPPOInnerTrainer, TrainerCfg
from efppo.rl.baseline_trainer import BaselineTrainer, BaselineTrainerCfg
from efppo.task.f110 import F1TenthWayPoint
from efppo.utils.logging import set_logger_format
from efppo.utils.ckpt_utils import load_ckpt_ez
from efppo.rl.replay_buffer import ReplayBuffer 



from jaxrl.jaxrl_trainer import JAXRLTrainer

def main(
    name: Annotated[str, typer.Option('', help="Name of the run.")],
    ckpt: Annotated[Optional[str], typer.Option('', help="ckpt path.")],
    seed: int = 123445,
):
    
    set_logger_format()

    n_history = 0
    if 'hist' in name:
        n_history = int(re.search(r"(\d+)hist", name).group(1))
    
    control_mode = None
    train_mode = 'control'
    if 'offpolicy' in name:
        control_mode = 'random+pursuit'
        train_mode = 'offpolicy'

    task = F1TenthWayPoint(n_history=n_history, control_mode = control_mode)

    contour_modes = [mode.value for mode in task.CONTOUR_MODES]
    if 'contour' in name:
        contour_modes = list(map(int, re.search(r'contour((\d+(_\d+)*)|\d+)', name).group(1).split('_')))
    
    
    if 'jaxrl' in name:
        load_from_path = None
        stamped_name = '_'.join([current_timestamp, str(sha)[-5:], name])
        
        if os.path.exists(name):
            load_from_path = name
            stamped_name = '_'.join([
                    current_timestamp, 
                    str(sha)[-5:], 
                    '_'.join([
                        os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(name)))), 
                        os.path.basename(os.path.dirname(name))
                        ])
                        ])
        
        trainer = JAXRLTrainer(name = stamped_name, seed = seed, load_from_path = load_from_path)     
        
        
            
        trainer.train()
    elif 'baseline' in name:
        trainer = BaselineTrainer(task) 
         
        trainer_cfg = BaselineTrainerCfg(n_iters=10_000_000, train_after = 1_000, train_every= 3, log_every=100, eval_every=100, ckpt_every=100, contour_modes = contour_modes, train_mode = train_mode)
         
        if os.path.exists(name): 
            with open(os.path.join(os.path.dirname(os.path.abspath(name)), 'cfg.pt'), 'rb') as fp:
                stamped_name = '_'.join([
                    current_timestamp, 
                    str(sha)[-5:], 
                    '_'.join([
                        os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(name)))), 
                        os.path.basename(os.path.dirname(name))
                        ])
                        ])
                cfg = pickle.load(fp)
                alg_cfg = cfg["alg_cfg"] 
                collect_cfg = cfg['collect_cfg']
                alg = alg_cfg.alg
                
                if 'sac' in name:
                    if 'disc' in name:
                        alg: Baseline = BaselineSACDisc.create(jr.PRNGKey(0), task, alg_cfg) 
                    else:
                        alg: Baseline = BaselineSAC.create(jr.PRNGKey(0), task, alg_cfg) 
                elif 'dqn' in name:
                    alg: Baseline = BaselineDQN.create(jr.PRNGKey(0), task, alg_cfg) 
                    if 'cbf' in name:
                        alg: BaselineDQNCBF = BaselineDQNCBF.create(jr.PRNGKey(0), task, alg_cfg) 
                 
                ckpt_dict = load_ckpt_ez(name, {"alg": alg})
                key1, key2 = jr.split(jr.PRNGKey(seed))
                alg = ckpt_dict["alg"]

                replay_buffer = ReplayBuffer.create(key=key1, capacity = 1e5)
                replay_buffer_path = os.path.dirname(os.path.dirname(os.path.dirname(name)))
                if os.path.exists(replay_buffer_path):
                    replay_buffer.load(replay_buffer_path)
                trainer.run(key2, alg, replay_buffer, collect_cfg, stamped_name, trainer_cfg)
        else:
            stamped_name = '_'.join([current_timestamp, str(sha)[-5:], name])
            alg_cfg, collect_cfg = efppo.run_config.f110.get(name)
            alg_cfg.train.n_batches = 1
            alg_cfg.train.bc_ratio = 0.
            if 'sac_bc' in name:
                alg_cfg.train.bc_ratio = 1.
            if 'ensemble' in name:
                alg_cfg.net.n_critics = 30
            if 'normalize' in name:
                alg_cfg.train.obs_stats_decay = 0.0
            trainer.train(jr.PRNGKey(seed), alg_cfg, collect_cfg, stamped_name, trainer_cfg, iteratively = True)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=False, default = None, help='name the run')
    parser.add_argument('--ckpt', type=str, required=False, default = None, help='Path to the ckpt folder') 
    args = parser.parse_args()
    
    main(args.name, args.ckpt)
    exit(0)
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)
        exit(0)
        
