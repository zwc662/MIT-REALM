from typing import (
    Any, 
    Callable, 
    List, 
    Tuple, 
    Optional
    )

from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import install_import_hook

import pickle

from sysid.src.constants import Logging_Level

import numpy as np


from sklearn.model_selection import train_test_split

import GPy
import safeopt


class SafeOpt_Learner:

    def __init__(
            self,
            random_key,
            load_path: Optional[str] = None,
            test_size: float = 0.5,
            num_samples: int = 1000,
            variance: float = 2,
            bounds: List[Tuple[float]] = [
                (-100, 100), 
                (-100, 100), 
                (-10, 10), 
                (-100, 100), 
                (-100, 100), 
                (-10, 10)
                ],
            swarm_type: str = 'maximizers'
            ):
        self.random_key = random_key 
        self.test_size = test_size
        self.num_samples = num_samples

        self.bounds = bounds
        
        ## objective function
        self.kernel = GPy.kern.RBF(input_dim=len(bounds), variance=variance, lengthscale=1.0, ARD=True)
        self.noise_var = 0.05 ** 2
        self.swarm_type = swarm_type
 
        self.likelihoods = jnp.array([]) 
        self.gp = GPy.models.GPRegression(np.zeros(1, len(bounds)), np.zeros(1, 1), self.kernel, noise_var=self.noise_var)  


    def downsample(self, ids, X, Y, logger):
        # Make predictions on all data
        
        assert X.shape[0] == Y.shape[0] == ids.shape[0]

        # Prepare test data 
        all_ids = jnp.arange(X.shape[0])

        num_samples = self.num_samples  

        sampled_X = None
        sampled_Y = None
        
 
        opt = safeopt.SafeOptSwarm(self.gp, 0., bounds=self.bounds, threshold=0.2)
 
        nxt_mean, nxt_std = opt.get_new_query_point(self.swarm_type)
       
        for i in ids:
            self.likelihoods.at[i].set(self.likelihoods[i] * 0.99 + \
                                           - (X[i] - nxt_mean).reshape(1, -1) @ jnp.diag(1 / (nxt_std**2)) @ (X[i] - nxt_mean).reshape(-1, 1) - \
                                            jnp.log(jnp.sqrt(jnp.abs(jnp.prod(nxt_std)))))
        likelihoods = self.likelihoods[ids]
        current_samples = 0


        # Non-replacement sample until collecting enough data
        while current_samples < num_samples:
            remaining_samples = num_samples - current_samples
            self.random_key, random_key = jax.random.split(self.random_key)
            
            if remaining_samples > len(all_ids):
                new_ids = jax.random.choice(random_key, all_ids, shape=(len(all_ids),), replace=False, p = likelihoods)
            elif remaining_samples == len(all_ids):
                new_ids = all_ids[:]
            else:
                new_ids = jax.random.choice(random_key, all_ids, shape=(remaining_samples,), replace=False, p = likelihoods)
            
            
            if sampled_X is None or sampled_Y is None:
                sampled_X = X[new_ids]
                sampled_Y = Y[new_ids]
            else:
                sampled_X = jnp.vstack((sampled_X, X[new_ids])) 
                sampled_Y = jnp.vstack((sampled_Y, Y[new_ids]))
                
            current_samples += len(new_ids)
       
        return sampled_X, sampled_Y


    def update(self, X, Y, logger):
        assert X.shape[0] > 20
        if self.likelihoods.shape[0] < X.shape[0]:
            self.likelihoods = jnp.concat([self.likelihoods, jnp.zeros([X.shape[0] - self.likelihoods.shape[0]])])

        ids_train, ids_test, X_train, X_test, y_train, y_test = train_test_split(
            np.arange(X.shape[0])[1:-10], 
            X[1:-10, 2:], 
            Y[1:-10], 
            test_size=self.test_size, 
            random_state=self.random_state
            )
        sampled_X, sampled_Y = self.downsample(ids_train, X_train, y_train, logger)
        logger.log(Logging_Level.STASH.value, f"{sampled_X.shape=}, {sampled_Y.shape=}")
        logger.log(Logging_Level.STASH.value, f"{sampled_X[0]=}, {sampled_Y[0]=}")

        self.gp = GPy.models.GPRegression(sampled_X, sampled_Y[:, 0:1], self.kernel, noise_var=self.noise_var) 
     
        opt = safeopt.SafeOptSwarm(self.gp, 0., bounds=self.bounds, threshold=0.2)
    
        means_x, vars_x = opt.gps[0].predict_noiseless(X_test)
        means_y, vars_y = opt.gps[1].predict_noiseless(X_test)
        
        # Compute the rMSE
        mse_x = jnp.sqrt(jnp.mean((means_x - y_test[:, 0]) ** 2))
        mse_y = jnp.sqrt(jnp.mean((means_y - y_test[:, 1]) ** 2))


 
        logger.log(Logging_Level.DEBUG.value, f"{y_test[0:10]=}")
        logger.log(Logging_Level.INFO.value, f'NN Model Mean Squared Error (x): {mse_x}')
        logger.log(Logging_Level.INFO.value, f'NN Model Mean Squared Error (y): {mse_y}')
         
        self.save_model()

        return {
            'mse_x': mse_x,
            'mse_y': mse_y
        }


            

