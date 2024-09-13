import ipdb
import jax.random as jr
import typer
from typing_extensions import Annotated

import sys

import git
repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha


from datetime import datetime

# Get current timestamp in yyy_mm_dd format
current_timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


import efppo.run_config.f110
from efppo.rl.efppo_inner_trainer import EFPPOInnerTrainer, TrainerCfg
from efppo.rl.baseline_trainer import BaselineTrainer, BaselineTrainerCfg

from efppo.task.f110 import F1TenthWayPoint
from efppo.utils.logging import set_logger_format
 
from jaxrl.jaxrl_trainer import JAXRLTrainer

def main(
    name: Annotated[str, typer.Option('', help="Name of the run.")],
    seed: int = 123445,
):
    set_logger_format()
    stamped_name = '_'.join([current_timestamp, str(sha)[-5:], name])

    if 'jaxrl' in name:
        trainer = JAXRLTrainer(name = stamped_name, seed = seed)     
        trainer.train()
    elif 'baseline' in name:
        task = F1TenthWayPoint()
        alg_cfg, collect_cfg = efppo.run_config.f110.get(name)
        alg_cfg.train.n_batches = 5
        alg_cfg.train.bc_ratio = 0.
        alg_cfg.net.n_critics = 30
        
        trainer = BaselineTrainer(task)
        #trainer = EFPPOInnerTrainer(task)
        trainer_cfg = BaselineTrainerCfg(n_iters=10_000_000, train_every= 10, log_every=10, eval_every=100, ckpt_every=100)
        trainer_cfg.train_every = 5
        trainer_cfg.log_every = 100
        trainer_cfg.eval_every = 100

        trainer.train(jr.PRNGKey(seed), alg_cfg, collect_cfg, stamped_name, trainer_cfg, iteratively = True)
    

if __name__ == "__main__":
    
    main(sys.argv[-1])
    exit(0)
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)
        exit(0)
        
