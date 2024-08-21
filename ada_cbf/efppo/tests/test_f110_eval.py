import pytest
import os
import sys
 
import functools as ft
import pathlib
import numpy as np
import jax
import jax.random as jr
import jax.numpy as jnp
import jax.lax as lax
import jax.tree_util as jtu

from efppo.utils.tfp import tfd
 
import matplotlib.pyplot as plt
import efppo.run_config.f110 as f110_config
from efppo.rl.collector import RolloutOutput, collect_single_env_mode
from efppo.rl.efppo_inner import EFPPOInner
from efppo.rl.rootfind_policy import Rootfinder, RootfindPolicy
from efppo.task.f110 import F1TenthWayPoint
from efppo.task.plotter import Plotter
from efppo.utils.ckpt_utils import get_run_dir_from_ckpt_path, load_ckpt_ez
from efppo.utils.jax_utils import jax2np, jax_vmap, merge01
from efppo.utils.logging import set_logger_format
from efppo.utils.path_utils import mkdir
from efppo.utils.tfp import tfd


 
@pytest.fixture
def get_pol():
    def help(obs_pol, state_z):
        return tfd.Bernoulli(logits=obs_pol[jnp.array([-2, -1])])
 
    return help



def test_eval():
    task = F1TenthWayPoint()
    # For prettier trajectories.
    # task.dt /= 2
    alg_cfg, collect_cfg = f110_config.get()
    alg: EFPPOInner = EFPPOInner.create(jr.PRNGKey(0), task, alg_cfg)
     
    for vgain in [0.1, 0.2, 0.5, 0.8, 1, 2]:
        rootfind_pol = lambda obs_pol, z: tfd.Normal(
            loc=(vgain, np.arctan(obs_pol[-1] / obs_pol[-2])) - obs_pol[0], 
            scale=1e-2 * np.ones(2)
            ) #[jnp.array([-2, -1])])
        # -----------------------------------------------------
        plot_dir = plot_dir = pathlib.Path(os.path.dirname(__file__))

        rollout_T = 10240

        bb_X, bb_Y, bb_x0 = jax2np(task.grid_contour())
        b1, b2 = bb_X.shape
        bb_z0 = np.full((b1, b2), alg.z_max)

        collect_fn = ft.partial(
            collect_single_env_mode,
            task,
            get_pol=rootfind_pol,
            disc_gamma=alg.disc_gamma,
            z_min=alg.z_min,
            z_max=alg.z_max,
            rollout_T=rollout_T,
        )
        print("Collecting rollouts...")
        bb_rollouts = [] #: list[list[RolloutOutput]] = []
        for i in range(bb_x0.shape[0]):
            bb_rollouts.append([])
            for j in range(bb_x0.shape[1]): 
                bb_rollouts[-1].append(collect_fn(task.reset() + 0 * bb_x0[i][j], bb_z0[i][j]))
            bb_rollouts[-1] = jtu.tree_map(lambda *x: jnp.stack(x), *bb_rollouts[-1])
            
        print("Done collecting rollouts.")
        bb_rollout = jtu.tree_map(lambda *x: jnp.stack(x), *bb_rollouts)
        ###############################3
        # Plot.
        rng = np.random.default_rng(seed=124122)
        plotter = Plotter(task)

        bbTp1_state = bb_rollout.Tp1_state
        bTp1_state = merge01(bbTp1_state)

        idxs = rng.choice(bTp1_state.shape[0], size=min(bTp1_state.shape[0], 100), replace=False)

        bTp1_state = bTp1_state[idxs]
        bb_h = np.max(bb_rollout.Th_h, axis=2)
        b_issafe = merge01(bb_h)[idxs] < 0

        figsize = np.array([2.8, 2.2])
        fig, ax = plt.subplots(figsize=figsize, dpi=500)
        fig = plotter.plot_traj(bTp1_state, multicolor=True, ax=ax)
        fig_path = plot_dir / f"eval_trajs_vgain_{vgain}.jpg"
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)

        # -----------------------------------------------
        # Plot trajectory in time.
        bTp1_state_safe = bTp1_state[b_issafe]
        fig = plotter.plot_traj2(bTp1_state_safe)
        fig_path = plot_dir / f"eval_traj_time_{vgain}.jpg"
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)
 
if __name__ == "__main__":
    pytest.main()