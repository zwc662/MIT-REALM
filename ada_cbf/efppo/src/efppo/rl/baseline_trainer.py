import pathlib
import time

from typing import Tuple, Union, List

import sys, os

import jax.random as jr
import jax.tree_util as jtu

import matplotlib.pyplot as plt
import numpy as np
from attrs import define
from loguru import logger
from matplotlib.colors import BoundaryNorm 

from enum import Enum

import wandb
from efppo.rl.collector import Collector, CollectorCfg
from efppo.rl.replay_buffer import ReplayBuffer, Experience
from efppo.rl.baseline import Baseline, BaselineSAC, BaselineSACDisc, BaselineDQN, EvalData
from efppo.task.plotter import Plotter
from efppo.task.task import Task
from efppo.utils.cfg_utils import Cfg
from efppo.utils.ckpt_utils import get_ckpt_manager_sync
from efppo.utils.jax_utils import jax2np, move_tree_to_cpu
from efppo.utils.path_utils import get_runs_dir, mkdir
from efppo.utils.register_cmaps import register_cmaps
from efppo.utils.rng import PRNGKey
from efppo.utils.wandb_utils import reorder_wandb_name

import pickle

sys.path.append(
    os.path.join(
        os.path.dirname(
            os.path.dirname(__file__)
        ), 'scripts/'
    )
)
#from eval_f110_wp import main as eval_main


# Enum mapping strings to classes
class BaselineEnum(Enum):
    SAC = BaselineSAC
    DISC_SAC = BaselineSACDisc
    DQN = BaselineDQN


@define
class BaselineTrainerCfg(Cfg):
    n_iters: int
    train_after: int
    log_every: int
    train_every: int
    eval_every: int
    ckpt_every: int

    ckpt_max_keep: int = 100

    contour_modes: List[int] = [1]
    contour_size: Tuple[int] = (10, 10)

    train_mode: str = 'online'
    

class BaselineTrainer:
    def __init__(self, task: Task):
        self.task = task
        self.plotter = Plotter(task) 
        register_cmaps()

    def plot_train(self, idx: int, plot_dir: pathlib.Path, data: Experience):
        
        # --------------------------------------------
        # Plot the trajectories.
        figsize = 1.5 * np.array([8, 6])
        Tp1_state, T_l = data.Tp1_state, data.T_l
        fig, ax = plt.subplots(figsize=figsize, dpi=500)
        figsize = 1.5 * np.array([8, 6])
        fig = self.plotter.plot_dots(states = Tp1_state, values = T_l)
        fig_path = mkdir(plot_dir / "phase") / "phase_{:08}_replaybuffer.jpg".format(idx)
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)


    def plot_eval(self, idx: int, plot_dir: pathlib.Path, data: EvalData, trainer_cfg: BaselineTrainerCfg):
        fig_opt = dict(layout="constrained", dpi=200)

        # --------------------------------------------
        # Plot rollouts
        nz, plot_batch_size, T, nx = data.z_eval_rollout.zbT_x.shape
        figsize = 1.5 * np.array([8, 6])
        fig, axes = plt.subplots(3, 3, figsize=figsize, **fig_opt)
        axes = axes.ravel().tolist()
        for grididx, ax in enumerate(axes[:nz]):
            z = data.z_zs[grididx]
            self.plotter.plot_traj(data.z_eval_rollout.zbT_x[grididx], multicolor=True, ax=ax)
            ax.set_title(f"z={z:.1f}")
        fig_path = mkdir(plot_dir / "phase") / "phase_{:08}.jpg".format(idx)
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)

        # --------------------------------------------
        # Contour of critic.
        for mode, eval_contour_val in data.z_eval_contours.items():
            #bb_X, bb_Y, _ = self.task.grid_contour(*trainer_cfg.contour_size, self.task.contour_modes[mode])
            zbb_X, zbb_Y = eval_contour_val.zbb_X, eval_contour_val.zbb_Y
            
            cmap = "rocket"     
            fig, axes = plt.subplots(3, 3, figsize=figsize, **fig_opt)
            axes = axes.ravel().tolist()
            for grididx, ax in enumerate(axes[:nz]):
                z = data.z_zs[grididx]
                cm = ax.contourf(zbb_X[grididx], zbb_Y[grididx], eval_contour_val.zbb_critic[grididx], levels=32, cmap=cmap)
                #self.task.setup_traj_plot(ax)
                fig.colorbar(cm, ax=ax)
                ax.set(xlabel='x', ylabel='y', title=f"z={z:.1f}")
            fig_path = mkdir(plot_dir / "critic") / "{:s}_{:08}.jpg".format(mode, idx)
            fig.savefig(fig_path, bbox_inches="tight")
            plt.close(fig)
 
            # --------------------------------------------
            # Contour of policy.
            # Use 'bwr' colormap and discretize it into task.n_actions colors (one for each integer action)
            cmap = plt.get_cmap('bwr', self.task.n_actions)  # discrete colors from the 'bwr' colormap
            # Create a norm to map values to the integers 0 to 10
            norm = BoundaryNorm(boundaries=np.arange(self.task.n_actions), ncolors=cmap.N)
            
            fig, axes = plt.subplots(3, 3, figsize=figsize, **fig_opt)
            axes = axes.ravel().tolist()
            for grididx, ax in enumerate(axes[:nz]):
                z = data.z_zs[grididx]
                cm = ax.contourf(zbb_X[grididx], zbb_Y[grididx], eval_contour_val.zbb_pol[grididx], levels=32, cmap=cmap, norm=norm)
                #self.task.setup_traj_plot(ax)
                fig.colorbar(cm, ax=ax, ticks=np.arange(self.task.n_actions))
                ax.set(xlabel='x', ylabel='y', title=f"z={z:.1f}")
            fig_path = mkdir(plot_dir / "policy") / "{:s}_{:08}.jpg".format(mode, idx)
            fig.savefig(fig_path, bbox_inches="tight")
            plt.close(fig)

            '''
            # --------------------------------------------
            # Contour of Vh.
            fig, axes = plt.subplots(3, 3, figsize=figsize, **fig_opt)
            axes = axes.ravel().tolist()
            for grididx, ax in enumerate(axes[:nz]):
                z = data.z_zs[grididx]
                cm = ax.contourf(bb_X, bb_Y, data.zbb_Vh[grididx], norm=CenteredNorm(), levels=32, cmap=cmap_Vh)
                self.task.setup_traj_plot(ax)
                fig.colorbar(cm, ax=ax)
                ax.set(xlabel=xlabel, ylabel=ylabel, title=f"z={z:.1f}")
            fig_path = mkdir(plot_dir / "Vh") / "Vh_{:08}.jpg".format(idx)
            fig.savefig(fig_path, bbox_inches="tight")
            plt.close(fig)
            '''

    def train(
        self, key: PRNGKey, alg_cfg: Baseline.Cfg, collect_cfg: CollectorCfg, wandb_name: str, trainer_cfg: BaselineTrainerCfg, iteratively: bool = True, 
    ):
        key0, key1, key2 = jr.split(key, 3)
        alg: Baseline = alg_cfg.alg.create(key0, self.task, alg_cfg) 
        replay_buffer = ReplayBuffer.create(key=key1, capacity = 1e5)
        self.run(key2, alg, replay_buffer, collect_cfg, wandb_name, trainer_cfg)
        
    def run(self, key: PRNGKey, alg: Baseline, replay_buffer: ReplayBuffer, collect_cfg: CollectorCfg, wandb_name: str, trainer_cfg: BaselineTrainerCfg, iteratively: bool = True):    
        task_name = self.task.name
        wandb_config = {"alg": alg.cfg.asdict(), "collect": collect_cfg.asdict(), "trainer": trainer_cfg.asdict()}
        wandb.init(project=f"baseline_{task_name}_inner", config=wandb_config, entity = 'zwc662') #, mode="disabled")
        wandb_run_name = reorder_wandb_name(wandb_name=wandb_name)

        run_dir = mkdir(get_runs_dir() / f"{task_name}_Baseline" / wandb_run_name)
        plot_dir = mkdir(run_dir / "plots")
        ckpt_dir = mkdir(run_dir / "ckpts")
        ckpt_manager = get_ckpt_manager_sync(ckpt_dir, max_to_keep=trainer_cfg.ckpt_max_keep)
 
        print(f"Initial data collection for {trainer_cfg.train_after} steps")
        collector: Collector = Collector.create(key, self.task, collect_cfg)
        collector, rollout = alg.collect_iteratively(collector, rollout_T = trainer_cfg.train_after, train_mode = trainer_cfg.train_mode)
        replay_buffer.insert(rollout)
        idx = 0
        for idx in range(trainer_cfg.n_iters):
            should_log = idx % trainer_cfg.log_every == 0
            should_eval = idx % trainer_cfg.eval_every == 0
            should_ckpt = idx % trainer_cfg.ckpt_every == 0

            t0 = time.time()

            print(f"Iteration {idx} / {trainer_cfg.n_iters}: Collecting ... ")
            collector, rollout = alg.collect_iteratively(collector, rollout_T = trainer_cfg.train_every, train_mode = trainer_cfg.train_mode)
            replay_buffer.insert(rollout)
            print(f"Iteration {idx} / {trainer_cfg.n_iters}: Updating ... ")
            t1 = time.time()
            alg, update_info = alg.update_iteratively(replay_buffer) #alg.update(replay_buffer) #
            t2 = time.time()
            #print(update_info)
            if should_log:
                if should_eval:
                    logger.info("time  |  collect: {:.3f} update: {:.3f}".format(t1 - t0, t2 - t1))
                log_dict = {f"train/{k}": v for k, v in update_info.items()}
                log_dict["time/collect"] = t1 - t0
                log_dict["time/update"] = t2 - t1
                wandb.log(log_dict, step=idx)

                loss_info = {k[5:]: float(v) for k, v in update_info.items() if k.startswith("Loss/")}
                logger.info(f"[{idx:8}]   {loss_info}")

            eval_rollout_T = collect_cfg.rollout_T
            if should_eval:
                if iteratively:
                    data = alg.eval_iteratively(self.task, eval_rollout_T, trainer_cfg.contour_modes, trainer_cfg.contour_size, train_mode = trainer_cfg.train_mode)
                    data = jtu.tree_map(np.array, data)
                else:
                    data = jax2np(alg.eval(eval_rollout_T))
                logger.info(f"[{idx:8}]   {data.info}")

                self.plot_eval(idx, plot_dir, data, trainer_cfg)
                log_dict = {f"eval/{k}": v for k, v in data.info.items()}
                wandb.log(log_dict, step=idx)

                #eval_main(alg = alg)

                # Remove the last trajectory in case it is dangling
                replay_buffer.truncate_from_right()
                # Reset the collector so that it samples from training maps
                collector = collector.reset()
                self.plot_train(idx, plot_dir, replay_buffer.experiences)
                
                

            if should_ckpt:
                ckpt_manager.save_ez(idx, {"alg": alg}) #, "alg_cfg": alg_cfg, "collect_cfg": collect_cfg})
                #ckpt_manager.save_ez(idx, {"policy": alg.policy, "Vl": alg.Vl, "Vh": alg.Vh})
                logger.info(f"Saved ckpt at {ckpt_dir}/{idx:08}/default/ !")

                with open(f'{ckpt_dir}/{idx:08}/cfg.pt', 'wb') as fp:
                    pickle.dump({"alg_cfg": alg.cfg, "collect_cfg": collect_cfg}, fp)
                
                replay_buffer.save(os.path.dirname(ckpt_dir))
                replay_buffer.truncate_from_right()

 
        # Save at the end.
        ckpt_manager.save_ez(idx, {"alg": alg})
        logger.info(f"Saved ckpt at {ckpt_dir}/{idx:8}/default/ !")
        with open(f'{ckpt_dir}/{idx:08}/cfg.pt', 'wb') as fp:
            pickle.dump({"alg_cfg": alg.cfg, "collect_cfg": collect_cfg}, fp)
        