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
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

import GPy
import safeopt



class Linear_Learner:

    def __init__(
            self,
            random_key,
            load_path: Optional[str] = None,
            random_state: float = 0,
            test_size: float = 0.5,
           
            ):
        self.random_key = random_key 
        self.random_state = random_state
        self.test_size = test_size
  
        self.safety = -8

        fun = None  
         
         
        self.model = LinearRegression()
     


    def update(self, X, Y, logger):
        assert X.shape[0] > 20
        if logger.level == Logging_Level.TEST.value:
            assert (Y[:, 0] == X[:, 2] + X[:, 3]).all()
            assert (Y[:, 1] == X[:, 2] - X[:, 3]).all()

        ids_train, ids_test, X_train, X_test, y_train, y_test = train_test_split(
            np.arange(X.shape[0])[1:-10], 
            X[1:-10, 2:], 
            Y[1:-10], 
            test_size=self.test_size, 
            random_state=self.random_state
            )
        
        if logger.level == Logging_Level.TEST.value:
            assert (y_train[:, 0] == X_train[:, 0] + X_train[:, 1]).all()
            assert (y_train[:, 1] == X_train[:, 0] - X_train[:, 1]).all()

       
        self.model.fit(X_train, y_train)
        preds_test = self.model.predict(X_test)
        mse_x = jnp.sqrt(jnp.mean((preds_test[:, 0] - y_test[:, 0]) ** 2))
        mse_y = jnp.sqrt(jnp.mean((preds_test[:, 1] - y_test[:, 1]) ** 2))


        logger.log(Logging_Level.INFO.value, f'Linear Model Mean Squared Error (x): {mse_x}')
        logger.log(Logging_Level.INFO.value, f'Linear Model Mean Squared Error (y): {mse_y}')
        logger.log(Logging_Level.TEST.value, f'Linear coef: {self.model.coef_}')
        

        return {
            'mse_x': mse_x,
            'mse_y': mse_y
        }


class Poly_Learner:

    def __init__(
            self,
            random_key,
            load_path: Optional[str] = None,
            random_state: float = 0,
            test_size: float = 0.5,
           
            ):
        self.random_key = random_key 
        self.random_state = random_state
        self.test_size = test_size
   
        # Transform features into polynomial features
        degree = 2
        self.poly = PolynomialFeatures(degree=degree)
         
         
        self.model = LinearRegression()
     


    def update(self, X, Y, logger):
        assert X.shape[0] > 20
        if logger.level == Logging_Level.TEST.value:
            assert (Y[:, 0] == X[:, 2] + X[:, 3]).all()
            assert (Y[:, 1] == - X[:, 2] - X[:, 3]).all()


        X_poly = self.poly.fit_transform(X[:, 2:])
        

        ids_train, ids_test, X_train, X_test, y_train, y_test = train_test_split(
            np.arange(X.shape[0])[1:-10], 
            X_poly[1:-10], 
            Y[1:-10], 
            test_size=self.test_size, 
            random_state=self.random_state
            )
        
        if logger.level == Logging_Level.TEST.value:
            assert (y_train[:, 0] == X_train[:, 0] + X_train[:, 1]).all()
            assert (y_train[:, 1] == - X_train[:, 0] - X_train[:, 1]).all()

        self.model.fit(X_train, y_train)
        preds_test = self.model.predict(X_test)
        mse_x = jnp.sqrt(jnp.mean((preds_test[:, 0] - y_test[:, 0]) ** 2))
        mse_y = jnp.sqrt(jnp.mean((preds_test[:, 1] - y_test[:, 1]) ** 2))


        logger.log(Logging_Level.INFO.value, f'SafeOPT Model Mean Squared Error (x): {mse_x}')
        logger.log(Logging_Level.INFO.value, f'SafeOPT Model Mean Squared Error (y): {mse_y}')
        

        return {
            'mse_x': mse_x,
            'mse_y': mse_y
        }


                          