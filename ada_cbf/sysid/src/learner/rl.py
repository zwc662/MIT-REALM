
from typing import List, Tuple, Optional, Any, Callable

import jax
import jax.numpy as jnp

from flax import linen as nn
from flax.training import train_state
import optax

from sysid.src.constants import Logging_Level

from sklearn.model_selection import train_test_split

import math

import pickle


@jax.jit
def train(carry, _):
    state, rng, batch, ens_grads = carry
    rng, new_rng = jax.random.split(rng)
    
    loss_fn = lambda params: jnp.mean(
        (state.apply_fn(params, batch['inputs'], training=True, rngs={'dropout': rng}) - batch['targets'])**2
        )
 
    value_and_grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = value_and_grad_fn(state.params)

    if len(ens_grads) == 0:
        ens_grads = grads
    else:
        ens_grads = jax.tree_map(
            lambda *grads_lst: jnp.sum(jnp.stack(grads_lst), axis = 0), *[ens_grads, grads])
    return (state, new_rng, batch, ens_grads), loss



@jax.jit
def eval(carry, _):
    state, rng, batch = carry
    rng, new_rng = jax.random.split(rng)
    preds =  state.apply_fn(state.params, batch['inputs'], training=True, rngs={'dropout': rng})
    return (state, new_rng, batch), preds



class NN_Learner:
    def __init__(self, 
                random_key,                
                learning_rate: float =1e-3, 
                hidden_sizes: Tuple[int] = (32, 32, 2),
                input_shape: Tuple[int]=(1,4),
                nn_file_path: Optional[str] = None,
                random_state: float = 0, 
                test_size: float = 0.5, 
                num_epochs: int = 10, 
                batch_size: int = 32, 
                num_samples: int = 50
                ):
        self.random_key = random_key
        model = MLP(hidden_sizes)
        if nn_file_path is None:
            random_key, self.random_key = jax.random.split(self.random_key)
            params = model.init(random_key, jnp.ones(input_shape))
            tx = optax.adam(learning_rate)
            self.state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
        else:
            self.state = self.load_model(model, nn_file_path)


        self.random_state = random_state
        self.test_size = test_size
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.num_data = self.batch_size * self.num_samples


    def downsample(self, X, Y, logger):
        # Make predictions on all data
        # Prepare test data 
        data_set = {'inputs': X, 'targets': Y}

        # Get testing results
        (_, self.random_key, _), predictions = jax.lax.scan(eval, (self.state, self.random_key, data_set), length=self.num_samples)

        logger.log(Logging_Level.DEBUG.value, f"{X.shape=}, {Y.shape=}")
        logger.log(Logging_Level.DEBUG.value, f"{predictions.shape=}")
        
        # Get the ensemble mean
        mean_preds = jnp.stack(predictions).mean(axis = 0)
        
        # Compute the rMSE
        mse = jnp.pow(1 - jnp.exp(-jnp.sqrt((mean_preds[:, 0] - Y[:, 0]) ** 2 + (mean_preds[:, 1] - Y[:, 1]) ** 2)), 0.99)

        assert X.shape[0] == Y.shape[0]
        all_ids = jnp.arange(X.shape[0])
        
        num_samples = self.num_samples * self.batch_size 

        sampled_X = None
        sampled_Y = None
        current_samples = 0
        
        while current_samples < num_samples:
            remaining_samples = num_samples - current_samples
            self.random_key, random_key = jax.random.split(self.random_key)
            
            if remaining_samples >= len(all_ids):
                new_ids= jax.random.choice(random_key, all_ids, shape=(len(all_ids),), replace=True, p = mse)
            else:
                new_ids = jax.random.choice(random_key, all_ids, shape=(remaining_samples,), replace=False, p = mse)
            
            
            if sampled_X is None or sampled_Y is None:
                sampled_X = X[new_ids]
                sampled_Y = Y[new_ids]
            else:
                logger.log(Logging_Level.DEBUG.value, f"{sampled_X.shape=}, {X[new_ids].shape=}")
                logger.log(Logging_Level.DEBUG.value, f"{sampled_Y.shape=}, {Y[new_ids].shape=}")
        
                sampled_X = jnp.vstack((sampled_X, X[new_ids])) 
                sampled_Y = jnp.vstack((sampled_Y, Y[new_ids]))
                
            current_samples += len(new_ids)
        
        # Concatenate all sampled parts into a single array
        return sampled_X, sampled_Y

        

    def update(self, X, Y, logger):

        assert X.shape[0] > 20


        X_train, X_test, y_train, y_test = train_test_split(X[1:-10, 2:], Y[1:-10], test_size=self.test_size, random_state=self.random_state)

        sampled_X, sampled_Y = self.downsample(X_train, y_train, logger)

        
        X_train = jnp.array(X_train)
        X_test = jnp.array(X_test)
        y_train = jnp.array(y_train)
        y_test = jnp.array(y_test)

        
        num_batches = math.ceil(len(X_train) / self.batch_size)
        logger.log(Logging_Level.STASH.value, f"{num_batches = }")
        

        for epoch in range(self.num_epochs):
            rand_key, self.random_key = jax.random.split(self.random_key)
            logger.log(Logging_Level.STASH.value, f"{X_train.shape[0] = }")
            # Get training batch data indexes
            batch_ids = jax.random.randint(rand_key, (num_batches, self.num_samples, self.batch_size), minval=0, maxval=X_train.shape[0])
            logger.log(Logging_Level.STASH.value, f"{batch_ids = }")

            for batch_id in range(num_batches):
                # Get training batch data
 
                batch = {
                    'inputs': jnp.stack([X_train[idx] for idx in batch_ids[batch_id]]),
                    'targets': jnp.stack([y_train[idx] for idx in batch_ids[batch_id]])
                }
        
                logger.log(Logging_Level.STASH.value, f"batch['inputs'].shape = {batch['inputs'].shape}")
                
                logger.log(Logging_Level.STASH.value, f"self.state.params = {list(self.state.params['params'])}")
                 
                # Get grads computed from each dropout
                grads = jax.tree_map(lambda param: jnp.zeros(param.shape), self.state.params) 
                
                (_, self.random_key, _, ens_grads), losses = jax.lax.scan(
                    train, (self.state, self.random_key, batch, grads), 
                    length=self.num_samples
                    )
            
                avg_loss = jnp.mean(losses)
                avg_grads = jax.tree_map(lambda grad: grad / self.num_samples, ens_grads)
                
                logger.log(Logging_Level.STASH.value, f"{ens_grads}")
                
                # Update with average grad
                self.state = self.state.apply_gradients(grads=avg_grads)
                logger.log(Logging_Level.INFO.value, f'Epoch: {epoch} | Batch: {batch_id} | Loss: {avg_loss}')

        # Prepare test data 
        test_batch = {'inputs': X_test, 'targets': y_test}

        # Get testing results
        (_, self.random_key, _), predictions = jax.lax.scan(eval, (self.state, self.random_key, test_batch), length=self.num_samples)

        logger.log(Logging_Level.DEBUG.value, f"{X_test.shape=}, {y_test.shape=}")
        logger.log(Logging_Level.DEBUG.value, f"{predictions.shape=}")
        
        # Get the ensemble mean
        mean_preds = jnp.stack(predictions).mean(axis = 0)
        
        # Compute the rMSE
        mse_x = jnp.sqrt(jnp.mean((mean_preds[:, 0] - y_test[:, 0]) ** 2))
        mse_y = jnp.sqrt(jnp.mean((mean_preds[:, 1] - y_test[:, 1]) ** 2))


        logger.log(Logging_Level.DEBUG.value, f"{mean_preds[0:10]=}")
        logger.log(Logging_Level.DEBUG.value, f"{y_test[0:10]=}")
        logger.log(Logging_Level.INFO.value, f'NN Model Mean Squared Error (x): {mse_x}')
        logger.log(Logging_Level.INFO.value, f'NN Model Mean Squared Error (y): {mse_y}')
         
        self.save_model()

        return {
            'mse_x': mse_x,
            'mse_y': mse_y
        }



    def save_model(self, filepath = 'nn.pt'):
        """Save model parameters to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.state.params, f)
        #logger.log(Logging_Level.INFO, f"Model parameters saved to {filepath}")

    def load_model(self, model, filepath = 'nn.pt'):
        """Load model parameters from a file."""
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
        self.state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(1e-3))
        #logger.log(Logging_Level.INFO, f"Model parameters loaded from {filepath}")
        


    


class MLP(nn.Module):
    hidden: List = (16, 16, 2)
 
    @nn.compact
    def __call__(self, x, training: bool = False):
        for hid in self.hidden[:-1]:
            x = nn.Dense(hid)(x)
            x = nn.relu(x)
            x = nn.Dropout(0.5)(x, deterministic=not training)
        x = nn.Dense(self.hidden[-1])(x)
        return x