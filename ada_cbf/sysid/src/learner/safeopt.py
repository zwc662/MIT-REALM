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
            random_state: float = 0,
            test_size: float = 0.5,
            num_samples: int = 1000,
            variance: float = 0.1,
            bounds: List[Tuple[float]] = [(-5, 5)] * (5 * 2 + 2),
            swarm_type: str = 'expanders'
            ):
        self.random_key = random_key 
        self.random_state = random_state
        self.test_size = test_size
        self.num_samples = num_samples

        self.bounds = bounds
        
        ## objective function
        self.kernel = GPy.kern.RBF(input_dim=len(self.bounds), variance=variance, lengthscale=1., ARD=True) 

        self.noise_var = 0.05 ** 2
        self.swarm_type = swarm_type
 
        self.likelihoods = jnp.array([[], []]) 

        self.safety = -8

        fun = None  
         
        self.gp = None # GPy.models.GPRegression(X = X0, Y = Y0, kernel = self.kernel, noise_var=self.noise_var)
        self.opt = None # safeopt.SafeOptSwarm(self.gp, self.safety, bounds=self.bounds, threshold=-10)
    
        self.performance_fn = lambda mses: -jnp.log(1.e3 + (jnp.sum(mses)) * 0.5)



    def update(self, X, Y, logger):
        assert X.shape[0] > 20
        if logger.level == Logging_Level.TEST.value:
            assert (Y[:, 0] == X[:, 4] + X[:, 5]).all()
            assert (Y[:, 1] == - X[:, 4] - X[:, 5]).all()

        ids_train, ids_test, X_train, X_test, y_train, y_test = train_test_split(
            np.arange(X.shape[0])[1:-10], 
            X[1:-10, 2:], 
            Y[1:-10], 
            test_size=self.test_size, 
            random_state=self.random_state
            )
        
        if logger.level == Logging_Level.TEST.value:
            assert (y_train[:, 0] == X_train[:, 2] + X_train[:, 3]).all()
            assert (y_train[:, 1] == - X_train[:, 2] - X_train[:, 3]).all()

        for i in range(10): 
            if self.gp is None and self.opt is None:
                X0 = jnp.zeros([1, X_train.shape[-1] * 2 + 2])
                Y0 = jnp.asarray([[-jnp.log(1.e3 + (jnp.sqrt(jnp.mean((y_train[:, 0]) ** 2)) + jnp.sqrt(jnp.mean((y_train[:, 1]) ** 2))) * 0.5)]])
                self.gp = GPy.models.GPRegression(X = X0, Y = Y0, kernel = self.kernel, noise_var=self.noise_var)
                self.opt = safeopt.SafeOptSwarm(self.gp, self.safety, bounds=self.bounds, threshold=-10)
                
            #try:
            wb = self.opt.optimize()
            #except RuntimeError:
            #    logger.log(Logging_Level.INFO.value, 'The safe set is empty. No change to the downsampler')
                
            #    wb = 0. * jnp.asarray([0, 0, 0., 0., 1, -1, 1, -1, 0, 0, 0, 0, 0, 0])
        
            logger.log(Logging_Level.STASH.value, f'{wb.shape=}')
            
            w = wb[:-2].reshape(X_train.shape[-1], 2)
            b = wb[-2:].reshape(1, 2)
            
            logger.log(Logging_Level.INFO.value, f'{w=}, {b=}')

            preds_train = X_train @ w + b

            #logger.log(Logging_Level.STASH.value, f'{y_train=}, {preds_train=}')

            # Compute the rMSE
            mse_x = jnp.sqrt(jnp.mean((preds_train[:, 0] - y_train[:, 0]) ** 2))
            mse_y = jnp.sqrt(jnp.mean((preds_train[:, 1] - y_train[:, 1]) ** 2))

            performance = self.performance_fn(jnp.asarray([mse_x, mse_y]))
            logger.log(Logging_Level.INFO.value, f'{performance=}')

            self.opt.add_new_data_point(wb, performance)
            
            #self.save_model()



        # Compute the rMSE
        preds_test = X_test @ w + b
        mse_x = jnp.sqrt(jnp.mean((preds_test[:, 0] - y_test[:, 0]) ** 2))
        mse_y = jnp.sqrt(jnp.mean((preds_test[:, 1] - y_test[:, 1]) ** 2))


        logger.log(Logging_Level.INFO.value, f'SafeOPT Model Mean Squared Error (x): {mse_x}')
        logger.log(Logging_Level.INFO.value, f'SafeOPT Model Mean Squared Error (y): {mse_y}')


        return {
            'mse_x': mse_x,
            'mse_y': mse_y
        }


            


class SafeOpt_Learner_Deprecated:

    def __init__(
            self,
            random_key,
            load_path: Optional[str] = None,
            random_state: float = 0,
            test_size: float = 0.5,
            num_samples: int = 1000,
            variance: float = 0.1,
            bounds: List[Tuple[float]] = [
                #(-100, 100),[] 
                #(-100, 100), 
                (-10, 10), 
                (-100, 100), 
                (-100, 100), 
                (-10, 10)
                ],
            swarm_type: str = 'expanders'
            ):
        self.random_key = random_key 
        self.random_state = random_state
        self.test_size = test_size
        self.num_samples = num_samples

        self.bounds = bounds
        
        ## objective function
        self.kernels = [
            GPy.kern.RBF(input_dim=len(bounds), variance=variance, lengthscale=1.0, ARD=True), 
            GPy.kern.RBF(input_dim=len(bounds), variance=variance, lengthscale=1.0, ARD=True)
        ]
        self.noise_var = 0.05 ** 2
        self.swarm_type = swarm_type
 
        self.likelihoods = jnp.array([[], []]) 
        self.gps = [
            GPy.models.GPRegression(X = np.zeros((1, len(bounds))), Y = np.zeros((1, 1)), kernel = self.kernels[0], noise_var=self.noise_var),  
            GPy.models.GPRegression(X = np.zeros((1, len(bounds))), Y = np.zeros((1, 1)), kernel = self.kernels[1], noise_var=self.noise_var)
        ] 
        
        self.nxt_means = jnp.array([[], []]) 
        self.nxt_stds = jnp.array([[], []]) 


    def downsample(self, ids, X, Y, logger):
        # Make predictions on all data
        
        assert X.shape[0] == Y.shape[0] == ids.shape[0]

        # Prepare test data 
        all_ids = jnp.arange(X.shape[0])

        num_samples = self.num_samples  

        sampled_Xs = []
        sampled_Ys = []
        
 
        opts = [safeopt.SafeOptSwarm(gp, 0., bounds=self.bounds, threshold=1) for gp in self.gps]

        for j, opt in enumerate(opts):
            try:
                #nxt_means, nxt_stds = opt.get_new_query_point(self.swarm_type)
                #logger.log(Logging_Level.DEBUG.value, f'{nxt_means.shape=}, {nxt_stds.shape=}')
                target_x = opt.optimize(True)
                logger.log(Logging_Level.DEBUG.value, f'{target_x.shape=}')

                for i in ids:
                    self.likelihoods.at[(j, i)].set(self.likelihoods[j, i] * 0.99 - jnp.mean((X[i] - target_x)**2))                          
                    #self.likelihoods.at[(j, i)].set(self.likelihoods[j, i] * 0.99 + \
                                                #- (X[i] - nxt_means).reshape(1, -1) @ jnp.diag(1 / (jnp.log(nxt_stds)**2)) @ (X[i] - nxt_means).reshape(-1, 1))
            except RuntimeError:
                logger.log(Logging_Level.WARNING.value, 'The safe set is empty. No change to the downsampler')
                        
            
            likelihoods = self.likelihoods[j, ids]
            
            # Non-replacement sample until collecting enough data
          
            sampled_X = None
            sampled_Y = None

            current_samples = 0

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

            sampled_Xs.append(sampled_X)
            sampled_Ys.append(sampled_Y)
                
            
       
        return sampled_Xs, sampled_Ys


    def update(self, X, Y, logger):
        assert X.shape[0] > 20
        if self.likelihoods.shape[1] < X.shape[0]:
            self.likelihoods = jnp.hstack((
                self.likelihoods, 
                jnp.zeros([2, X.shape[0] - self.likelihoods.shape[0]])
                )
                )
            logger.log(Logging_Level.DEBUG.value, f"{self.likelihoods.shape=}")

        ids_train, ids_test, X_train, X_test, y_train, y_test = train_test_split(
            np.arange(X.shape[0])[1:-10], 
            X[1:-10, 2:], 
            Y[1:-10], 
            test_size=self.test_size, 
            random_state=self.random_state
            )
        
        sampled_Xs, sampled_Ys = self.downsample(ids_train, X_train, y_train, logger)
        logger.log(Logging_Level.DEBUG.value, f"{sampled_Xs[0].shape=}, {sampled_Ys[0].shape=}")
        logger.log(Logging_Level.DEBUG.value, f"{sampled_Xs[0][0]=}, {sampled_Ys[0][0]=}")

        self.gps = [
            GPy.models.GPRegression(sampled_Xs[0], sampled_Ys[0][:, 0:1], self.kernels[0], noise_var=self.noise_var),
            GPy.models.GPRegression(sampled_Xs[1], sampled_Ys[1][:, 1:2], self.kernels[1], noise_var=self.noise_var) 
        ]
     
        
        means_x, vars_x = self.gps[0].predict_noiseless(X_test)
        means_y, vars_y = self.gps[1].predict_noiseless(X_test)
        
        # Compute the rMSE
        mse_x = jnp.sqrt(jnp.mean((means_x - y_test[:, 0]) ** 2))
        mse_y = jnp.sqrt(jnp.mean((means_y - y_test[:, 1]) ** 2))


 
        logger.log(Logging_Level.DEBUG.value, f"{y_test[0:10]=}")
        logger.log(Logging_Level.INFO.value, f'NN Model Mean Squared Error (x): {mse_x}')
        logger.log(Logging_Level.INFO.value, f'NN Model Mean Squared Error (y): {mse_y}')
         
        #self.save_model()

        return {
            'mse_x': mse_x,
            'mse_y': mse_y
        }


            
