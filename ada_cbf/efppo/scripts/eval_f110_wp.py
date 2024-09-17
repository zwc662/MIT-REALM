import os
import sys

from typing import Optional, Annotated, Union
 
import argparse
import functools as ft
import pathlib
import numpy as np
import jax
import jax.random as jr
import jax.numpy as jnp
import jax.lax as lax
import jax.tree_util as jtu
import pickle
import h5py
import re 

from efppo.utils.tfp import tfd
 
import matplotlib.pyplot as plt 
import efppo.run_config.f110 as f110_config
from efppo.rl.collector import RolloutOutput, collect_single_env_mode
from efppo.rl.efppo_inner import EFPPOInner
from efppo.rl.baseline import Baseline, BaselineSAC, BaselineSACDisc, BaselineDQN
from efppo.rl.rootfind_policy import Rootfinder, RootfindPolicy
from efppo.task.f110 import F1TenthWayPoint
from efppo.task.plotter import Plotter
from efppo.utils.ckpt_utils import get_run_dir_from_ckpt_path, load_ckpt_ez
from efppo.utils.jax_utils import jax2np, jax_vmap, merge01
from efppo.utils.logging import set_logger_format
from efppo.utils.path_utils import mkdir
from efppo.utils.tfp import tfd
from efppo.utils.cfg_utils import Recursive_Update 

from jaxrl.agents import (AWACLearner, DDPGLearner, REDQLearner, SACLearner,
                          SACV1Learner)

import git
repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha


from datetime import datetime

# Get current timestamp in yyy_mm_dd format
current_timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
stamped_name = '_'.join([current_timestamp, str(sha)[-5:]])


def main(
        alg: Optional[Union[Baseline, EFPPOInner]] = None, 
        ckpt_path: Optional[pathlib.Path] = None,
        render: bool = False,
        control_mode: Optional[str] = None,
        **kwargs):
    set_logger_format()
    n_history = 0
    if ckpt_path is not None and 'hist' in str(ckpt_path):
        n_history = int(re.search(r"(\d+)hist",str(ckpt_path)).group(1))
    task = F1TenthWayPoint(control_mode = control_mode, n_history = n_history)
      
    # Plot.
    rng = np.random.default_rng(seed=124122)
    plotter = Plotter(task)
    
    plot_dir = mkdir(pathlib.Path(os.path.join(os.path.dirname(__file__), 'plots')))

 
    
    
    rollout_T = 1000
    disc_gamma=0.99
    z_min=0
    z_max=3
    
    rootfind_pol = lambda obs_pol, z: 0
    if alg is not None:
        if isinstance(alg, EFPPOInner):
            rootfind = Rootfinder(alg.Vh.apply, alg.z_min, alg.z_max, h_tgt=-0.70)
            rootfind_pol = lambda obs, z: RootfindPolicy(alg.policy.apply, rootfind)(obs, z).mode()
        elif isinstance(alg, Baseline):
            rootfind_pol = lambda obs, z: alg.policy.apply(obs, z).mode()
    elif ckpt_path is not None:
        plot_dir = mkdir(pathlib.Path(os.path.join(os.path.dirname(ckpt_path), 'plots')))
        if 'F1TenthWayPoint_JAXRL' in ckpt_path:
            if 'sac' in ckpt_path:
                alg_cls = SACV1Learner 
            elif 'ql' in ckpt_path and 'ql' in ckpt_path:
                alg_cls = REDQLearner
            obs_example = np.zeros([task.nobs])
            act_example = np.zeros([task.nu])
            alg = alg_cls(0, str(ckpt_path), obs_example, act_example, **kwargs) 
            rootfind_pol = lambda obs, *args, **kwargs: alg.sample_actions(obs)
        elif 'F1TenthWayPoint_Baseline' in ckpt_path:       
            with open(os.path.join(os.path.dirname(ckpt_path), 'cfg.pt'), 'rb') as fp:
                cfg = pickle.load(fp)
                alg_cfg = cfg["alg_cfg"]
                collect_cfg = cfg['collect_cfg']
                if 'sac' in ckpt_path:
                    if 'disc' in ckpt_path:
                        alg: Baseline = BaselineSACDisc.create(jr.PRNGKey(0), task, alg_cfg) 
                    else:
                        alg: Baseline = BaselineSAC.create(jr.PRNGKey(0), task, alg_cfg) 
                elif 'dqn' in ckpt_path:
                    alg: Baseline = BaselineDQN.create(jr.PRNGKey(0), task, alg_cfg) 
            print(f'Load from {ckpt_path}')
            ckpt_dict = load_ckpt_ez(ckpt_path, {"alg": alg})
            alg = ckpt_dict["alg"]
            rootfind_pol = lambda obs, z: alg.policy.apply(obs, z).mode()
            rollout_T = collect_cfg.max_T
            disc_gamma=alg.disc_gamma
            z_min=alg.cfg.train.z_min
            z_max=alg.cfg.train.z_max

        elif 'F1TenthWayPoint_Inner' in ckpt_path:
            alg_cfg, collect_cfg = f110_config.get()
            alg: EFPPOInner = EFPPOInner.create(jr.PRNGKey(0), task, alg_cfg)
            print(f'Load from {ckpt_path}')
            ckpt_dict = load_ckpt_ez(ckpt_path, {"alg": alg})
            alg = ckpt_dict["alg"]
            rootfind = Rootfinder(alg.Vh.apply, alg.z_min, alg.z_max, h_tgt=-0.70)
            rootfind_pol = lambda obs, z: RootfindPolicy(alg.policy.apply, rootfind)(obs, z).mode()
            rollout_T = collect_cfg.max_T
            disc_gamma=alg.disc_gamma
            z_min=alg.cfg.train.z_min
            z_max=alg.cfg.train.z_max

    bb_X, bb_Y, bb_x0 = jax2np(task.grid_contour(n_xs=10, n_ys=10))
    b1, b2 = bb_X.shape
    bb_z0 = np.full((b1, b2), z_max)
  
    
    collect_fn = ft.partial(
        collect_single_env_mode,
        task,
        get_pol=rootfind_pol,
        disc_gamma=disc_gamma,
        z_min=z_min,
        z_max=z_max,
        rollout_T=rollout_T,
        verbose=False,
        soft_reset = False
    )
    print("Collecting rollouts...")
    bb_rollouts: list[list[RolloutOutput]] = []
    for i in range(bb_x0.shape[0]):
        bb_rollouts.append([])
        for j in range(bb_x0.shape[1]): 
            state_0, hist_0 = task.reset(mode=f"eval{'+render' if render else ''}", init_dist = np.random.normal(loc = 0, scale = 0.1, size = (3)))
            print('Initialization state coord', (i, j), f'from map {task.cur_map_name}')
            rollout = collect_fn(state_0.flatten(), bb_z0[i][j])
            #print(rollout.Tp1_state.shape, rollout.Tp1_obs.shape, rollout.Tp1_z.shape, rollout.T_control.shape, rollout.T_l.shape, rollout.Th_h.shape)
            bb_rollouts[-1].append(rollout) 
        bb_rollouts[-1] = jtu.tree_map(lambda *x: jnp.stack(x), *bb_rollouts[-1])
        
        print("Done collecting rollouts.")
        bb_rollout = jtu.tree_map(lambda *x: jnp.stack(x), *bb_rollouts)
        
        ## 1. Draw values along the trajectories
        Tp1_state = merge01(merge01(bb_rollout.Tp1_state)[:, :-1])
        Tp1_obs = merge01(merge01(bb_rollout.Tp1_obs)[:, :-1])
        Tp1_z = merge01(merge01(bb_rollout.Tp1_z)[:, :-1])

        T_target_critics_all = jax.vmap(alg.target_critic.apply)(Tp1_obs, Tp1_z).reshape(-1, alg.cfg.net.n_critics, task.n_actions)
        T_target_critics = jnp.max(T_target_critics_all, axis = -1).reshape(-1, alg.cfg.net.n_critics)
        
        value_file_name = f"{task.cur_map_name}{('_' + control_mode) if control_mode is not None else ''}"

        states_path, critics_path = plot_dir / f"{value_file_name}_states.h5", plot_dir / f"{value_file_name}_critics.h5"
        
        with h5py.File(states_path, 'a' if os.path.exists(states_path) else 'w' ) as fp:
            fp.create_dataset(f'{value_file_name}_{stamped_name}_state_{i}', data = Tp1_state)
        with h5py.File(critics_path, 'a' if os.path.exists(critics_path) else 'w') as fp:
            fp.create_dataset(f'{value_file_name}_{stamped_name}_critics_{i}', data = T_target_critics)


        plotter.task.PLOT_2D_INDXS = [
                    plotter.task.STATE_X, 
                    plotter.task.STATE_Y
                    ]
        
        with h5py.File(states_path, 'r') as fp:
            Tp1_state = np.concatenate([fp[k] for k in list(fp.keys())])
        with h5py.File(critics_path, 'r') as fp:
            T_target_critics = np.concatenate([fp[k] for k in list(fp.keys())])
        
        
        for func in [jnp.min, jnp.max, jnp.mean]:
            fig = plotter.plot_dots(states = Tp1_state, values = func(T_target_critics, axis = -1))
            fig_path = plot_dir / f"{value_file_name}_eval_value_{func.__name__}.jpg"
            fig.savefig(fig_path, bbox_inches="tight")
            print(f"Saved figure at {fig_path}")
            plt.close(fig)

    ## 2. Draw first, last waypoints, and trajectories
    bb_rollout = jtu.tree_map(lambda *x: jnp.stack(x), *bb_rollouts)
    ###############################3
        
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
        fig_path = plot_dir / f"{task.cur_map_name}_eval_{label}.jpg"
        fig.savefig(fig_path, bbox_inches="tight")
        print(f"Saved figure at {fig_path}")
        plt.close(fig)

    ## 3. Draw h and l values along safe and unsafe trajectories
    b_h = merge01(np.max(bb_rollout.Th_h, axis=(2, 3)))

    b_issafe = (b_h <= -1e-3).astype(float).reshape(-1).astype('float64')
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
        fig_path = plot_dir / f"{task.cur_map_name}_eval_safe_traj_time{('_' + kwargs['idx']) if 'idx' in kwargs else ''}.jpg"
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)


    b_isunsafe = (b_h > 1e-3).astype(float).reshape(-1).astype('float64')
    if b_isunsafe.sum() >= 1.:
        p = b_isunsafe / b_isunsafe.sum() 
        print(p)
        unsafe_idxs = rng.choice(bTp1_state.shape[0], size=min(b_isunsafe.sum().astype(int).item(), 5), replace=False, p = p)
        #b_isunsafe = bTh_h[unsafe_idxs] 
        # -----------------------------------------------
        # Plot unsafe trajectory in time.
        bTp1_state_unsafe = bTp1_state[unsafe_idxs]
        bT_l_unsafe = bT_l[unsafe_idxs]
        bTh_h_unsafe = bTh_h[unsafe_idxs] 
        fig= plotter.plot_traj3(bTp1_state_unsafe, bTh_h_unsafe, bT_l_unsafe)
        fig_path = plot_dir / f"{task.cur_map_name}_eval_unsafe_traj_time{('_' + kwargs['idx']) if 'idx' in kwargs else ''}.jpg"
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', type=str, required=False, default = None, help='select efppo or baseline')
    parser.add_argument('--ckpt', type=str, required=False, default = None, help='Path to the ckpt folder')
    parser.add_argument('--work', type=int, required=False, default = 0, help='Path to the map without extensions')
    parser.add_argument('--params', type=int, required=False, default = 0, help='Path to the map without extensions')
    parser.add_argument('--pursuit', action='store_true', help='use pursuit planner to override any control input')
    parser.add_argument('--random', action='store_true', help='use random control to override any control input')
    parser.add_argument('--render', action='store_true', help='render track')
    args = parser.parse_args()

    control_mode = None
    if args.pursuit:
        control_mode = 'pursuit'
    if args.random:
        control_mode = 'random'

    main(alg = args.alg, ckpt_path = args.ckpt, render = args.render, control_mode = control_mode)
    #with ipdb.launch_ipdb_on_exception():
    #    typer.run(main)
