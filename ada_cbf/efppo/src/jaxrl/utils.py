from typing import Optional

from efppo.task.f110 import F1TenthWayPoint

from jaxrl import wrappers


def make_env(env_name: str,
             seed: int,
             add_episode_monitor: bool = True,
             action_repeat: int = 1,
             frame_stack: int = 1,
             from_pixels: bool = False,
             pixels_only: bool = True,
             image_size: int = 84,
             sticky: bool = False,
             gray_scale: bool = False,
             flatten: bool = True) -> gym.Env:
    # Check if the env is in gym.
    train_env = F1TenthWayPoint()
    eval_env = F1TenthWayPoint()

    action_space = ()
    env.observation_space.seed(seed)

    return env
