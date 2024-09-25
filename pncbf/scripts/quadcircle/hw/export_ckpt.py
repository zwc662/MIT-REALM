import pathlib

import ipdb
import typer
from loguru import logger

import run_config.int_avoid.quadcircle_cfg
import run_config.int_avoid.segway_cfg
import run_config.nclf.pend_cfg
import run_config.nclf_pol.pend_cfg
from pncbf.dyn.quadcircle import QuadCircle
from pncbf.ncbf.int_avoid import IntAvoid
from pncbf.utils.ckpt_utils import get_id_from_ckpt, get_run_path_from_ckpt, load_ckpt_with_step, save_ckpt
from pncbf.utils.logging import set_logger_format
from pncbf.utils.path_utils import mkdir


def main(ckpt_path: pathlib.Path):
    set_logger_format()
    task = QuadCircle()

    CFG = run_config.int_avoid.quadcircle_cfg.get(0)
    nom_pol = task.nom_pol_vf
    alg: IntAvoid = IntAvoid.create(0, task, CFG.alg_cfg, nom_pol)
    alg, ckpt_path = load_ckpt_with_step(alg, ckpt_path)
    logger.info("Loaded ckpt from {}!".format(ckpt_path))
    cid = get_id_from_ckpt(ckpt_path)

    run_path = get_run_path_from_ckpt(ckpt_path)
    export_dir = mkdir(run_path / "exports")
    ckpt_path = export_dir / f"ckpt{cid}"

    model = {"params": alg.Vh.params, "act": CFG.alg_cfg.act, "hids": CFG.alg_cfg.hids}
    save_ckpt(model, ckpt_path)
    logger.info("Exported ckpt to {}!".format(ckpt_path))


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)
