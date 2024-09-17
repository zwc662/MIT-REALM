import pytest
import os
import sys
 
import numpy as np
import jax
import jax.numpy as jnp
import jax.lax as lax


from efppo.task.f110 import F1TenthWayPoint




def test_train():
    sys.path.append(
        os.path.join(
            os.path.dirname(
                os.path.dirname(__file__)
            ), 'scripts/'
        )
    )
    from train_f110_wp import main as train_main
    train_main(name = 'test_f110_train_jaxrl_sac')

''' 
def test_vmap():
    num_steps = 10
    
    def run_task(seed):
        # Reset the task to get initial state
        task = F1TenthWayPoint(seed=seed)
        obs, reward, done, info = task.reset()

        # Define the step function for jax.lax.scan
        def step_fn(carry, _):
            obs, reward, done, info = carry
            control = jnp.ones(2)  # Generate control input within the step_fn
            obs, reward, done, info = jax.lax.stop_gradient(task.step(obs, control))
            return (obs, reward, done, info), (obs, reward, done, info)

        # Run the scan to perform multiple steps
        dummy_controls = jnp.arange(num_steps)  # Just a placeholder to iterate over
        final_state, (obs_seq, reward_seq, done_seq, info_seq) = jax.lax.scan(step_fn, (obs, reward, done, info), dummy_controls)
        return final_state
    
    seeds = np.arange(1, 100, 10)
    
    # Use vmap to apply run_task across all task instances
    all_final_states = jax.vmap(run_task)(seeds)
    
    # Assertions can be based on expected values
    assert all_final_states.shape == (len(seeds), 4)  # 4 corresponds to (obs, reward, done, info)
'''
if __name__ == "__main__":
    pytest.main()