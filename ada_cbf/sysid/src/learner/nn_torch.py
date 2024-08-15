
from typing import List, Tuple, Optional, Any, Callable

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.modules import Linear

 
from sysid.src.constants import Logging_Level 



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

from sklearn.compose import TransformedTargetRegressor


import math

import pickle

import torch
import torch.nn.functional as F

#define _l2normalization
def _l2normalize(v, eps=1e-12):
    return v / (torch.norm(v) + eps)

def max_singular_value(W, u=None, Ip=1):
    """
    power iteration for weight parameter
    """
    #xp = W.data
    if not Ip >= 1:
        raise ValueError("Power iteration should be a positive integer")
    if u is None:
        u = torch.FloatTensor(1, W.size(0)).normal_(0, 1).cuda()
    _u = u
    for _ in range(Ip):
        _v = _l2normalize(torch.matmul(_u, W.data), eps=1e-12)
        _u = _l2normalize(torch.matmul(_v, torch.transpose(W.data, 0, 1)), eps=1e-12)
    sigma = torch.sum(F.linear(_u, torch.transpose(W.data, 0, 1)) * _v)
    return sigma, _u


class SNLinear(Linear):
    r"""Applies a linear transformation to the incoming data: :math:`y = Ax + b`
       Args:
           in_features: size of each input sample
           out_features: size of each output sample
           bias: If set to False, the layer will not learn an additive bias.
               Default: ``True``
       Shape:
           - Input: :math:`(N, *, in\_features)` where :math:`*` means any number of
             additional dimensions
           - Output: :math:`(N, *, out\_features)` where all but the last dimension
             are the same shape as the input.
       Attributes:
           weight: the learnable weights of the module of shape
               `(out_features x in_features)`
           bias:   the learnable bias of the module of shape `(out_features)`

           W(Tensor): Spectrally normalized weight

           u (Tensor): the right largest singular value of W.
       """
    def __init__(self, in_features, out_features, bias=True):
        super(SNLinear, self).__init__(in_features, out_features, bias)
        self.register_buffer('u', torch.Tensor(1, out_features).normal_())

    @property
    def W_(self):
        w_mat = self.weight.view(self.weight.size(0), -1)
        sigma, _u = max_singular_value(w_mat, self.u)
        self.u.copy_(_u)
        return self.weight / sigma

# Define the MLP class
class SNMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(SNMLP, self).__init__()
        # Define the layers
        self.fcs = nn.Sequential(
            SNLinear(input_size, hidden_sizes[0]),
            SNLinear(hidden_sizes[0], hidden_sizes[1]),
            SNLinear(hidden_sizes[1], output_size)
            )
        
    def forward(self, x):
        # Define the forward pass
        for fc in self.fcs[:-1]:
            x = F.relu(fc(x)) 
        y = F.tanh(self.fcs[-1](x))
        return y

class SN_NN_Learner:
    def __init__(self,   
                random_key,     
                learning_rate: float =1e-3, 
                hidden_sizes: Tuple[int] = (32, 32, 2),
                input_shape: Tuple[int]=(1,5),
                nn_file_path: Optional[str] = None,
                random_state: float = 0, 
                test_size: float = 0.5, 
                num_epochs: int = 10, 
                batch_size: int = 32, 
                num_samples: int = 50
                ): 

        self.model = SNMLP(input_shape[-1], hidden_sizes[:-1], hidden_sizes[-1]) 
        self.test_size = test_size
        self.random_state = random_state

        # Loss function and optimizer
        self.criterion = nn.MSELoss(reduction='none')  # To get MSE for each output dim
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.num_epochs = num_epochs

        # Choose device based on whether CUDA is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Define scalers
        self.scaler_X = MinMaxScaler(feature_range=(-1, 1))
        self.scaler_y = MinMaxScaler(feature_range=(-1, 1))

 
    def update(self, X, Y, logger):
        assert X.shape[0] > 20

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X[1:-10, 2:], Y[1:-10], test_size=self.test_size, random_state=self.random_state)

        # Fit the scalers and transform the data
        X_train = self.scaler_X.fit_transform(X_train)
        X_test = self.scaler_X.transform(X_test)
        y_train = self.scaler_y.fit_transform(y_train)
        y_test = self.scaler_y.transform(y_test)

        # Convert data to PyTorch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        y_test = torch.tensor(y_test, dtype=torch.float32).to(self.device)

        # Training loop
        for epoch in range(self.num_epochs):
            self.model.train()

            self.optimizer.zero_grad()
            outputs = self.model(X_train)
            loss = self.criterion(outputs, y_train).mean()  # Calculate MSE for each output dim separately
            loss.backward()
            self.optimizer.step()

            # Log the training loss for each output dimension
            logger.log(Logging_Level.DEBUG.value, f"Epoch [{epoch + 1}/{self.num_epochs}], Loss: {loss.detach().cpu().numpy()}")

        # Evaluation on the test set
        self.model.eval()
        with torch.no_grad():
            test_outputs = self.model(X_test)
            test_loss = self.criterion(test_outputs, y_test).mean(dim=0)  # MSE for each output dim

        # Log the test MSE for each output dimension
        logger.log(Logging_Level.DEBUG.value, f"Test MSE: {test_loss.detach().cpu().numpy()}")

        
        logger.log(Logging_Level.INFO.value, f'NN Model Mean Squared Error (x): {test_loss[0]}')
        logger.log(Logging_Level.INFO.value, f'NN Model Mean Squared Error (y): {test_loss[1]}')
         
        #self.save_model()

        return {
            'mse_x': test_loss[0],
            'mse_y': test_loss[1]
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
        #self.state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(1e-3))
        #logger.log(Logging_Level.INFO, f"Model parameters loaded from {filepath}")
        
    def eval(self, Xs, logger):
        # Inverse scale the predictions to the original scale
        with torch.no_grad():
            ys = self.model(torch.tensor(Xs, dtype=torch.float32).to(self.device))
            preds = self.scaler_y.inverse_transform(ys.cpu().numpy())
            return preds, 0 * preds

      