import os
import sys

from typing import Optional, Annotated
 
import argparse
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




def main(
        alg: Optional[EFPPOInner] = None, 
        ckpt_path: Optional[pathlib.Path] = None,
        render: bool = False,
        pursuit: bool = False,
        **kwargs):
    set_logger_format()

    plot_dir = pathlib.Path(os.path.dirname(__file__))
    if ckpt_path is not None:
        #plot_dir = get_run_dir_from_ckpt_path(ckpt_path)
        plot_dir = mkdir(plot_dir / str(ckpt_path).split('runs/')[-1].split('/ckpts')[0])
  
    task = F1TenthWayPoint(control_mode = 'pursuit' if pursuit else '')
    
    # For prettier trajectories.
    # task.dt /= 2
    alg_cfg, collect_cfg = f110_config.get()
    #collect_cfg.max_T = 2048
    if alg is None:
        alg: EFPPOInner = EFPPOInner.create(jr.PRNGKey(0), task, alg_cfg)
     
    for vgain in ([1, 0.2, 0.5, 2] if ckpt_path is None else [1]):
   
        steer_fn = lambda obs_pol: np.arctan(obs_pol[-1] / obs_pol[-2]) - obs_pol[task.OBS_YAW]
        rootfind_pol = lambda obs_pol, z: tfd.Normal(
            loc=(steer_fn (obs_pol), vgain), 
            scale=(np.pi * 0.5 * 0.7, 0.1)
            ) #[jnp.array([-2, -1])])
        # -----------------------------------------------------
        if ckpt_path is not None: 
            print(f'Load from {ckpt_path}')
            ckpt_dict = load_ckpt_ez(ckpt_path, {"alg": alg})
            alg = ckpt_dict["alg"]

        rootfind = Rootfinder(alg.Vh.apply, alg.z_min, alg.z_max, h_tgt=-0.70)
        rootfind_pol = RootfindPolicy(alg.policy.apply, rootfind)
        
        
            
        rollout_T = collect_cfg.max_T

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
        bb_rollouts: list[list[RolloutOutput]] = []
        for i in range(bb_x0.shape[0]):
            bb_rollouts.append([])
            for j in range(bb_x0.shape[1]): 
                print('Initialization state coord', (i, j))
                state_0 = None
                if (i, j) == (0, 0):
                    state_0 = task.reset(mode=f"eval{'+render' if render else ''}", random_map = True)
                else:
                    state_0 = task.reset(mode=f"eval{'+render' if render else ''}") 
                state_0 += 0 * bb_x0[i][j]
                rollout = collect_fn(state_0, bb_z0[i][j])
                print(rollout.Tp1_state.shape, rollout.Tp1_obs.shape, rollout.Tp1_z.shape, rollout.T_control.shape, rollout.T_l.shape, rollout.Th_h.shape)
                bb_rollouts[-1].append(rollout) 
            bb_rollouts[-1] = jtu.tree_map(lambda *x: jnp.stack(x), *bb_rollouts[-1])
            
        print("Done collecting rollouts.")
        bb_rollout = jtu.tree_map(lambda *x: jnp.stack(x), *bb_rollouts)
        ###############################3
        # Plot.
        rng = np.random.default_rng(seed=124122)
        plotter = Plotter(task)

        bbTp1_state = bb_rollout.Tp1_state
        bTp1_state = merge01(bbTp1_state)
        
        bbT_l = bb_rollout.T_l
        bT_l = merge01(bbT_l)

        bbTh_h = bb_rollout.Th_h
        bTh_h = merge01(bbTh_h)

        
        

        figsize = np.array([2.8, 2.2])
        for label in ['fst_wps', 'lst_wps', 'trajs']:
            if label == 'fst_wps':
                plotter.task.PLOT_2D_INDXS = [
                    plotter.task.STATE_FST_LAD, 
                    plotter.task.STATE_FST_LAD + 1
                    ]
            elif label == 'lst_wps':
                plotter.task.PLOT_2D_INDXS = [
                    plotter.task.STATE_FST_LAD + (plotter.task.conf.work.nlad - 1) * 2, 
                    plotter.task.STATE_FST_LAD + 1 + (plotter.task.conf.work.nlad - 1) * 2
                    ] 
            elif label == 'trajs':
                plotter.task.PLOT_2D_INDXS = [
                    plotter.task.STATE_X, 
                    plotter.task.STATE_Y
                    ]
            fig, ax = plt.subplots(figsize=figsize, dpi=500)
            fig = plotter.plot_traj(bTp1_state, multicolor=True, ax=ax)
            fig_path = plot_dir / f"eval_{label}_vgain_{vgain}.jpg"
            fig.savefig(fig_path, bbox_inches="tight")
            print(f"Saved figure at {fig_path}")
            plt.close(fig)
        

        b_h = merge01(np.max(bb_rollout.Th_h, axis=(2, 3)))

        b_issafe = (b_h <= 0).astype(float).reshape(-1).astype('float64')
        if b_issafe.sum() >= 1.:
            p = b_issafe / b_issafe.sum() 
            safe_idxs = rng.choice(bTp1_state.shape[0], size=min(b_issafe.sum().astype(int).item(), 5), replace=False, p = p)
            #b_issafe = bTh_h[safe_idxs] 
            # -----------------------------------------------
            # Plot safe trajectory in time.
            bTp1_state_safe = bTp1_state[safe_idxs]
            bT_l_safe = bT_l[safe_idxs]
            bTh_h_safe = bTh_h[safe_idxs] 
            fig = plotter.plot_traj3(bTp1_state_safe, bTh_h_safe, bT_l_safe)  
            fig_path = plot_dir / f"eval_safe_traj_time_{vgain}{('_' + kwargs['idx']) if 'idx' in kwargs else ''}.jpg"
            fig.savefig(fig_path, bbox_inches="tight")
            plt.close(fig)


        b_isunsafe = (b_h > 0).astype(float).reshape(-1).astype('float64')
        if b_isunsafe.sum() >= 1.:
            p = b_isunsafe / b_isunsafe.sum() 
            unsafe_idxs = rng.choice(bTp1_state.shape[0], size=min(b_isunsafe.sum().astype(int).item(), 5), replace=False, p = p)
            #b_isunsafe = bTh_h[unsafe_idxs] 
            # -----------------------------------------------
            # Plot unsafe trajectory in time.
            bTp1_state_unsafe = bTp1_state[unsafe_idxs]
            bT_l_unsafe = bT_l[unsafe_idxs]
            bTh_h_unsafe = bTh_h[unsafe_idxs] 
            fig = plotter.plot_traj3(bTp1_state_unsafe, bTh_h_unsafe, bT_l_unsafe)
            fig_path = plot_dir / f"eval_unsafe_traj_time_{vgain}{('_' + kwargs['idx']) if 'idx' in kwargs else ''}.jpg"
            fig.savefig(fig_path, bbox_inches="tight")
            plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=False, default = None, help='Path to the ckpt folder')
    parser.add_argument('--work', type=int, required=False, default = 0, help='Path to the map without extensions')
    parser.add_argument('--params', type=int, required=False, default = 0, help='Path to the map without extensions')
    parser.add_argument('--pursuit', action='store_true', help='use pursuit planner to override any control input')
    parser.add_argument('--render', action='store_true', help='render track')
    args = parser.parse_args()

    main(ckpt_path = args.ckpt, render = args.render, pursuit = args.pursuit)
    #with ipdb.launch_ipdb_on_exception():
    #    typer.run(main)
