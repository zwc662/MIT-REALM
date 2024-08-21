import ipdb
import jax.random as jr
import typer

import efppo.run_config.f110
from efppo.rl.efppo_inner_trainer import EFPPOInnerTrainer, TrainerCfg
from efppo.task.f110 import F1TenthWayPoint
from efppo.utils.logging import set_logger_format


def main(
    name: str = typer.Option(None, help="Name of the run."),
    seed: int = 123445,
):
    set_logger_format()
    task = F1TenthWayPoint()
    alg_cfg, collect_cfg = efppo.run_config.f110.get()
    trainer = EFPPOInnerTrainer(task)
    trainer_cfg = TrainerCfg(n_iters=10_000_000, log_every=100, eval_every=2_000, ckpt_every=10_000)
    trainer.train(jr.PRNGKey(seed), alg_cfg, collect_cfg, name, trainer_cfg, iteratively = True)


if __name__ == "__main__":
    main()
    exit(0)
    
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)
