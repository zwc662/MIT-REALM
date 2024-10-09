import multiprocessing as mp
import pathlib
import pickle
import time

import ipdb
import jax
import numpy as np
from loguru import logger
from mpire import WorkerPool
from rich.progress import MofNCompleteColumn, Progress, TaskProgressColumn, TimeElapsedColumn

from clfrl.dyn_cs.doubleint_cs import DoubleIntCS
from clfrl.mpc.mpc import MPCCfg, MPCResult, mpc_sim, mpc_sim_worker, mpc_worker_init
from clfrl.utils.compare_ci import CIData
from clfrl.utils.jax_utils import jax_jit, merge01, rep_vmap, tree_cat, tree_unmerge01
from clfrl.utils.paths import get_script_plot_dir


def main():
    plot_dir = get_script_plot_dir()

    n_nodes = 30
    dt = 0.025
    cfg = MPCCfg(n_nodes, dt, cost_reg=1e-3, mpc_T=70)

    task = DoubleIntCS()
    nom_pol = task.task.nom_pol_osc
    setup_idx = 0

    bb_x, bb_Xs, bb_Ys = task.task.get_paper_ci_x0()

    # n_cpu = mp.cpu_count()
    # n_cpu = 12
    n_cpu = 8
    # n_cpu = 6
    # n_cpu = 4

    bb_shape = bb_x.shape[:2]
    b_x = merge01(bb_x)
    mb_x = np.array_split(b_x, n_cpu)

    pbar = Progress(*Progress.get_default_columns(), MofNCompleteColumn(), TimeElapsedColumn())
    pbar.start()
    pbar_tasks = [pbar.add_task("W {:2}".format(ii), total=len(mb_x[ii])) for ii in range(n_cpu)]

    m = mp.Manager()
    queue: mp.Queue = m.Queue()

    with jax.transfer_guard("disallow"):
        with WorkerPool(n_jobs=n_cpu, shared_objects=queue, start_method="spawn") as pool:
            async_results = [
                pool.apply_async(mpc_sim_worker, args=(wid, task, b_x, nom_pol, cfg)) for wid, b_x in enumerate(mb_x)
            ]

            while not all(async_result.ready() for async_result in async_results):
                for _ in range(queue.qsize()):
                    wid, n_completed = queue.get()
                    pbar.update(pbar_tasks[wid], advance=n_completed)
                time.sleep(0.1)

            results = [async_result.get() for async_result in async_results]

    pbar.stop()
    b_results = tree_cat(results, axis=0)
    bb_results: MPCResult = tree_unmerge01(b_results, bb_shape)

    # Eval h on the traj.
    bbS_x = bb_results.S_x
    bbSh_h = jax_jit(rep_vmap(task.task.h_components, rep=3))(bbS_x)

    # Save results.
    ci_data = CIData(
        "MPC", task.task.name, setup_idx, bbS_x, bb_results.S_u, bbSh_h, bb_Xs, bb_Ys, notes={"mpc_T": cfg.mpc_T}
    )
    pkl_path = pathlib.Path(plot_dir / "mpc_dbint_T{}.pkl".format(cfg.mpc_T))
    with open(pkl_path, "wb") as f:
        pickle.dump(ci_data, f)

    logger.info("Saved to {}!".format(pkl_path))


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
