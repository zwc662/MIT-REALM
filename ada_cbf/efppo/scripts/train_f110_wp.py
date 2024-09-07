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
current_timestamp = datetime.now().strftime("%Y_%m_%d")


import efppo.run_config.f110
from efppo.rl.efppo_inner_trainer import EFPPOInnerTrainer, TrainerCfg
from efppo.rl.baseline_trainer import BaselineTrainer, BaselineTrainerCfg

from efppo.task.f110 import F1TenthWayPoint
from efppo.utils.logging import set_logger_format
 


def main(
    name: Annotated[str, typer.Option('', help="Name of the run.")],
    seed: int = 123445,
):
    set_logger_format()
    task = F1TenthWayPoint()
    alg_cfg, collect_cfg = efppo.run_config.f110.get(name)
    
    trainer = BaselineTrainer(task)
    #trainer = EFPPOInnerTrainer(task)
    trainer_cfg = BaselineTrainerCfg(n_iters=10_000_000, log_every=10, eval_every=10, ckpt_every=10)
    stamped_name = '_'.join([str(sha)[-5:], current_timestamp, name])

    trainer.train(jr.PRNGKey(seed), alg_cfg, collect_cfg, stamped_name, trainer_cfg, iteratively = True)
 

if __name__ == "__main__":
    
    main(sys.argv[-1])
    exit(0)
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)
        exit(0)
        
