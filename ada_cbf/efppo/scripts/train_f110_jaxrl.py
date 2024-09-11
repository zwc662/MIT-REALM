import ipdb
import jax.random as jr
import typer
from typing_extensions import Annotated

from argparse import Namespace

import sys

import git
repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha


from datetime import datetime

# Get current timestamp in yyy_mm_dd format
current_timestamp = datetime.now().strftime("%Y_%m_%d")


from jaxrl.jaxrl_trainer import JAXRLTrainer
from efppo.task.f110 import F1TenthWayPoint
from efppo.utils.logging import set_logger_format
 


def main(
    name: Annotated[str, typer.Option('', help="Name of the run.")],
    seed: int = 123445,
):
    set_logger_format()
    stamped_name = '_'.join([str(sha)[-5:], current_timestamp, name])
    trainer = JAXRLTrainer(name = stamped_name, seed = seed)     
    trainer.train()
 

if __name__ == "__main__":
    main(sys.argv[-1])
    exit(0)
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)
        exit(0)
        
