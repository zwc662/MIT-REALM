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
    sys.path.append(
        os.path.join(
            os.path.dirname(
                os.path.dirname(__file__)
            ), 'scripts/'
        )
    )
    from eval_f110_wp import main as eval_main
    ckpt_path = "/Users/weichaozhou/Workspace/MIT-REALM/ada_cbf/efppo/runs/F1TenthWayPoint_Baseline/0131-2024_09_13_18_02_28_7d833_baseline_disc_sac_ensemble_10acts/ckpts/00000200/default"
    eval_main(None, ckpt_path = ckpt_path, render = False, control_mode = None) #pathlib.Path(os.path.dirname(__file__)))

if __name__ == "__main__":
    pytest.main()