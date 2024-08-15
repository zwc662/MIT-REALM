
from typing import List, Tuple, Optional, Any, Callable

# Enable Float64 for more stable matrix inversions.
from jax import config

from functools import partial

#config.update("jax_enable_x64", True)

import jax 
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import install_import_hook

import pickle

from sysid.src.constants import Logging_Level

import numpy as np
#from docs.examples.utils import clean_legend


 
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler 

from sklearn.compose import TransformedTargetRegressor


class GP_Learner:

    def __init__(
            self, 
            random_key,  
            gp_file_path: Optional[str] = None,
            random_state: float = 0,
            test_size: float = 0.5,
            num_samples: int = 1000
            ):
        self.random_key = random_key
        self.random_state = random_state
        kernel = DotProduct() + WhiteKernel() 

        scaler_X = MinMaxScaler(feature_range=(-1, 1))
        scaler_y = MinMaxScaler(feature_range=(-1, 1))
         
        model =  GaussianProcessRegressor(kernel=kernel, random_state=self.random_state)
        #self.model = make_pipeline(StandardScaler(), model)


        pipeline = Pipeline([
            ('scaler', scaler_X),
            ('regress', model)
        ])
        self.model = TransformedTargetRegressor(regressor=pipeline, transformer=scaler_y)
            

        self.Ds = [] 

        if gp_file_path is not None:
            self.load_model(gp_file_path)
   
        self.num_samples = num_samples
        self.test_size = test_size
        self.likelihoods = jnp.array([])
 
      
    def eval(self, Xs, logger = None):    
        means_lst, vars_lst = [], []
        for i, (X, y) in enumerate(self.Ds):
            
            opt_posterior = self.model.fit(X, y)
            logger.log(Logging_Level.STASH.value, f'get posterior with dim {i}') 
            logger.log(Logging_Level.STASH.value, f'{X[0]=}')
        
            means = opt_posterior.predict(Xs, return_std=False)
            
        
            means_lst.append(means)
            #vars_lst.append(vars)

        return jnp.stack(means_lst, axis=-1).reshape(-1, 2), jnp.ones((means_lst[0].shape[0], len(self.Ds))).reshape(-1, 2) #jnp.stack(vars_lst, axis=-1)
    
    
      
    def downsample(self, ids, X, Y, logger):
        # Make predictions on all data
        
        assert X.shape[0] == Y.shape[0] == ids.shape[0]

        # Prepare test data 
        all_ids = jnp.arange(X.shape[0])

        num_samples = self.num_samples  

        sampled_X = None
        sampled_Y = None
 
        if len(self.Ds) > 0:
            # Get testing results
            
            #logger.log(Logging_Level.DEBUG.value, f"{X.shape=}, {Y.shape=}")
           
            # Compute the rMSE
            means, vars = self.eval(X, logger)
            for i, (means_, vars_, y) in enumerate(zip(means, vars, Y)):
                logger.log(Logging_Level.STASH.value, f"{means_.shape=}, {vars_.shape=}, {y.shape=}")
                logger.log(Logging_Level.STASH.value, f"{(y - means_).reshape(1, -1).shape=}")
                logger.log(Logging_Level.STASH.value, f"{jnp.diag(1 / vars_).shape=}")
                logger.log(Logging_Level.STASH.value, f"{(y - means_).reshape(-1, 1).shape=}")
                logger.log(Logging_Level.STASH.value, f"{jnp.log(jnp.sqrt(jnp.abs(jnp.prod(vars_))))=}")

                likelihood = ((y - means_).reshape(1, -1) @ jnp.diag(1 / vars_) @ (y - means_).reshape(-1, 1) - \
                    jnp.log(jnp.sqrt(jnp.abs(jnp.prod(vars_))))).item()
                #print(f"{self.likelihoods[i]=}, {likelihood=}")
                self.likelihoods.at[i].set(self.likelihoods[i] * 0.99 + likelihood)
                
                logger.log(Logging_Level.DEBUG.value, f"{self.likelihoods[i]=}")

        # Reverse ordering and normalizing the sampling likelihoods
        sel_likelihoods = self.likelihoods[ids]
        rev_likelihoods = jnp.min(sel_likelihoods) - sel_likelihoods
        rev_likelihoods -= 1 # Avoid devided by 0
        norm_rev_likelihoods = rev_likelihoods / jnp.sum(rev_likelihoods)
        

        current_samples = 0


        # Non-replacement sample until collecting enough data
        while current_samples < num_samples:
            remaining_samples = num_samples - current_samples
            self.random_key, random_key = jax.random.split(self.random_key)
            
            if remaining_samples > len(all_ids):
                new_ids = jax.random.choice(random_key, all_ids, shape=(len(all_ids),), replace=False, p = norm_rev_likelihoods)
            elif remaining_samples == len(all_ids):
                new_ids = all_ids[:]
            else:
                new_ids = jax.random.choice(random_key, all_ids, shape=(remaining_samples,), replace=False, p = norm_rev_likelihoods)
            
            
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

        self.Ds = [(sampled_X, sampled_Y[:, i:i+1]) for i in range(sampled_Y.shape[1])]

        # Concatenate all sampled parts into a single array

        logger.log(Logging_Level.STASH.value, f"{type(self.Ds[-1])=}")
       
        means, vars = self.eval(X_test, logger)

        print(f"{X_test[0]=}, {y_test[0]=}, {means[0]=}")
        
        # Compute the rMSE
        mse_x = jnp.sqrt(jnp.mean((means[:, 0] - y_test[:, 0]) ** 2))
        mse_y = jnp.sqrt(jnp.mean((means[:, 1] - y_test[:, 1]) ** 2))


        logger.log(Logging_Level.DEBUG.value, f"{means[0:10]=}")
        logger.log(Logging_Level.DEBUG.value, f"{y_test[0:10]=}")
 
        #print(f'NN Model Mean Squared Error (x): {mse_x}')
        #print(f'NN Model Mean Squared Error (y): {mse_y}')
        
        
        self.save_model()

        return {
            'mse_x': mse_x,
            'mse_y': mse_y
        }





    def save_model(self, filepath = 'gp.pt'):
        """Save model parameters to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.Ds, f)
        #logger.log(Logging_Level.INFO, f"Model parameters saved to {filepath}")

    def load_model(self, filepath = 'gp.pt'):
        """Load model parameters from a file."""
        with open(filepath, 'rb') as f:
            self.Ds = pickle.load(f)
        #logger.log(Logging_Level.INFO, f"Model parameters loaded from {filepath}")
        


 
class GPX_Learner:

    def __init__(
            self, 
            random_key,  
            gp_file_path: Optional[str] = None,
            random_state: float = 0,
            test_size: float = 0.5,
            num_samples: int = 1000
            ):
        self.random_key = random_key
        
        kernel = gpx.kernels.RBF()
        meanf = gpx.mean_functions.Zero()
        self.prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
       
        self.Ds = []
        self.posteriors = []

        if gp_file_path is not None:
            self.load_model(gp_file_path)
    
        self.random_state = random_state
        self.num_samples = num_samples
        self.test_size = test_size
        self.likelihoods = jnp.array([])
        

    
    def get_posterior(self, D: Any):
        likelihood = gpx.likelihoods.Gaussian(num_datapoints=D.n)
        posterior = self.prior * likelihood

        negative_mll = gpx.objectives.ConjugateMLL(negative=True)
        #negative_mll = jax.jit(negative_mll)

        opt_posterior, history = gpx.fit_scipy(
            model=posterior,
            objective=negative_mll,
            train_data=D,
        )
        return opt_posterior
    
    def eval(self, Xs, logger):    
        means_lst, vars_lst = [], []
        for i, D in enumerate(self.Ds):
            opt_posterior = self.get_posterior(D)
            logger.log(Logging_Level.DEBUG.value, f'get posterior with dim {i}')
            logger.log(Logging_Level.DEBUG.value, f'{Xs[0]}')
            logger.log(Logging_Level.DEBUG.value, f'{D=}')
        
            means, vars = [], []
            for x in Xs:
                latent_dist = opt_posterior.predict(x.reshape(1, -1), train_data=D)
                posterior_dist = opt_posterior.likelihood(latent_dist)
                mean = posterior_dist.mean() 
                var = posterior_dist.variance()
                means.append(mean)
                vars.append(var)

            means_lst.append(means)
            vars_lst.append(vars)

        return jnp.asarray(means_lst).reshape(-1, len(self.Ds)), \
            jnp.asarray(vars_lst).reshape(-1, len(self.Ds))

 
    
      
    def downsample(self, ids, X, Y, logger):
        # Make predictions on all data
        
        assert X.shape[0] == Y.shape[0] == ids.shape[0]

        # Prepare test data 
        all_ids = jnp.arange(X.shape[0])

        num_samples = self.num_samples  

        sampled_X = None
        sampled_Y = None
 
        if len(self.Ds) > 0:
            # Get testing results
            
            logger.log(Logging_Level.DEBUG.value, f"{X.shape=}, {Y.shape=}")
           
            # Compute the rMSE
            means, vars = self.eval(X, logger)
            for i, (means_, vars_, y) in enumerate(zip(means, vars, Y)):
                self.likelihoods.at[i].set(self.likelihoods[i] * 0.99 + \
                                           - (y - means_).reshape(1, -1) @ jnp.diag(1 / vars_) @ (y - means_).reshape(-1, 1) - \
                                            jnp.log(jnp.sqrt(jnp.abs(jnp.prod(vars_)))))
                logger.log(Logging_Level.DEBUG.value, f"{self.likelihoods[i]=}")

        # Reverse ordering and normalizing the sampling likelihoods
        sel_likelihoods = self.likelihoods[ids]
        rev_likelihoods = jnp.min(sel_likelihoods) - sel_likelihoods
        rev_likelihoods -= 1 # Avoid devided by 0
        norm_rev_likelihoods = rev_likelihoods / jnp.sum(rev_likelihoods)
        

        current_samples = 0


        # Non-replacement sample until collecting enough data
        while current_samples < num_samples:
            remaining_samples = num_samples - current_samples
            self.random_key, random_key = jax.random.split(self.random_key)
            
            if remaining_samples > len(all_ids):
                new_ids = jax.random.choice(random_key, all_ids, shape=(len(all_ids),), replace=False, p = norm_rev_likelihoods)
            elif remaining_samples == len(all_ids):
                new_ids = all_ids[:]
            else:
                new_ids = jax.random.choice(random_key, all_ids, shape=(remaining_samples,), replace=False, p = norm_rev_likelihoods)
            
            
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

        self.Ds = [gpx.Dataset(sampled_X, sampled_Y[:, i:i+1]) for i in range(sampled_Y.shape[1])]

        # Concatenate all sampled parts into a single array

        logger.log(Logging_Level.STASH.value, f"{type(self.Ds[-1])=}")
       
        means, vars = self.eval(X_test, logger)
        
        # Compute the rMSE
        mse_x = jnp.sqrt(jnp.mean((means[:, 0] - y_test[:, 0]) ** 2))
        mse_y = jnp.sqrt(jnp.mean((means[:, 1] - y_test[:, 1]) ** 2))


        logger.log(Logging_Level.DEBUG.value, f"{means[0:10]=}")
        logger.log(Logging_Level.DEBUG.value, f"{y_test[0:10]=}")
        logger.log(Logging_Level.INFO.value, f'NN Model Mean Squared Error (x): {mse_x}')
        logger.log(Logging_Level.INFO.value, f'NN Model Mean Squared Error (y): {mse_y}')
         
        self.save_model()

        return {
            'mse_x': mse_x,
            'mse_y': mse_y
        }





    def save_model(self, filepath = 'gp.pt'):
        """Save model parameters to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.Ds, f)
        #logger.log(Logging_Level.INFO, f"Model parameters saved to {filepath}")

    def load_model(self, filepath = 'gp.pt'):
        """Load model parameters from a file."""
        with open(filepath, 'rb') as f:
            self.Ds = pickle.load(f)
        #logger.log(Logging_Level.INFO, f"Model parameters loaded from {filepath}")
        

