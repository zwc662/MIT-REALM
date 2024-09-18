import ipdb
import jax.random as jr
import typer
from typing_extensions import Annotated

import sys
import os

import re

import git
repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha

import pickle 

from datetime import datetime

# Get current timestamp in yyy_mm_dd format
current_timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


import efppo.run_config.f110
from efppo.rl.baseline import Baseline, BaselineSAC, BaselineSACDisc, BaselineDQN
from efppo.rl.efppo_inner_trainer import EFPPOInnerTrainer, TrainerCfg
from efppo.rl.baseline_trainer import BaselineTrainer, BaselineTrainerCfg
from efppo.task.f110 import F1TenthWayPoint
from efppo.utils.logging import set_logger_format
from efppo.utils.ckpt_utils import load_ckpt_ez
from efppo.rl.replay_buffer import ReplayBuffer 

from jaxrl.jaxrl_trainer import JAXRLTrainer

def main(
    name: Annotated[str, typer.Option('', help="Name of the run.")],
    seed: int = 123445,
):
    
    set_logger_format()
    
    if 'jaxrl' in name:
        stamped_name = '_'.join([current_timestamp, str(sha)[-5:], name])
        trainer = JAXRLTrainer(name = stamped_name, seed = seed)     
        trainer.train()
    elif 'baseline' in name:
        n_history = 0
        if 'hist' in name:
            n_history = int(re.search(r"(\d+)hist", name).group(1))
        
        task = F1TenthWayPoint(n_history=n_history)
        trainer = BaselineTrainer(task) 
        trainer_cfg = BaselineTrainerCfg(n_iters=10_000_000, train_after = 1_000, train_every= 3, log_every=100, eval_every=100, ckpt_every=100)
         
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
                if 'sac' in name:
                    if 'disc' in name:
                        alg: Baseline = BaselineSACDisc.create(jr.PRNGKey(0), task, alg_cfg) 
                    else:
                        alg: Baseline = BaselineSAC.create(jr.PRNGKey(0), task, alg_cfg) 
                elif 'dqn' in name:
                    alg: Baseline = BaselineDQN.create(jr.PRNGKey(0), task, alg_cfg) 
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
    
    main(sys.argv[-1])
    exit(0)
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)
        exit(0)
        
