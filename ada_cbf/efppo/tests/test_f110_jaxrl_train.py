import pytest
import os
import sys
 
import numpy as np
import jax
import jax.numpy as jnp
import jax.lax as lax


from jaxrl.jaxrl_trainer import JAXRLTrainer




def test_train():
    sys.path.append(
        os.path.join(
            os.path.dirname(
                os.path.dirname(__file__)
            ), 'scripts/'
        )
    )
    trainer = JAXRLTrainer(name = 'test_f110_train_jaxrl_sac_30')
    trainer.train()
