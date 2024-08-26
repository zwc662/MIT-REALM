import datetime
import pathlib
from typing import Any

import numpy as np

import jax
import attrs
import orbax
import orbax.checkpoint
from attrs import asdict
from flax.training import orbax_utils
from flax import struct
from orbax.checkpoint import CheckpointManager 


def save_ckpt_ez(save_path: pathlib.Path, item: Any):
    ckpter = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(item)
    ckpter.save(save_path, item, save_args=save_args)


def load_ckpt_ez(load_path: pathlib.Path, item: Any):
    ckpter = orbax.checkpoint.PyTreeCheckpointer()
    
    restored_item = {}
    for k, v in item.items():
        if isinstance(v, struct.PyTreeNode):
            restored_item[k] = jax.tree_util.tree_map(lambda _: orbax.checkpoint.RestoreArgs(restore_type=np.ndarray), v)            
    
    return ckpter.restore(load_path, args = orbax.checkpoint.args.PyTreeRestore(item = restored_item))

    restored_item = ckpter.metadata(load_path)
    locally_sharded = lambda x: jax.device_put(x, jax.sharding.SingleDeviceSharding(jax.local_devices()[0]))
    restored_item = jax.tree_util.tree_map(locally_sharded, item)
    #restored_item = jax.tree_util.tree_map(
    #   orbax.checkpoint.utils.to_shape_dtype_struct, restored_item
    #)
    restored_item = ckpter.restore(load_path, item=restored_item)
    print(restored_item)
    
    
    for k, v in restored_item.items():
        if k not in item:
            del restored_item[k]
    return restored_item

class WrappedCkptManager(CheckpointManager):
    def save_ez(self, step: int, items: Any):
        if isinstance(items, dict):
            # Replacce all attrs dataclasses with their dict equivalents.
            items_ = {}
            for k, v in items.items():
                if attrs.has(v):
                    v = asdict(v)
                items_[k] = v
            items = items_

        save_args = orbax_utils.save_args_from_target(items)
        return self.save(step, items, save_kwargs={"save_args": save_args})


def get_ckpt_manager_sync(ckpt_dir: pathlib.Path, max_to_keep: int = 50, minutes: float = 5):
    options = orbax.checkpoint.CheckpointManagerOptions(
        max_to_keep=max_to_keep,
        keep_time_interval=datetime.timedelta(minutes=minutes),
        create=True,
        step_format_fixed_length=8,
    )
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    ckpt_manager = WrappedCkptManager(ckpt_dir, orbax_checkpointer, options)
    return ckpt_manager


def get_run_dir_from_ckpt_path(ckpt_path: pathlib.Path):
    # 0006-curio-eleva/ckpts/00020000/default/
    return ckpt_path.parent.parent.parent 
