
from abc import ABC, abstractmethod, abstractproperty
from typing import Optional, Tuple

from gcbfplus.utils.typing import Action, Params, PRNGKey, Array
from gcbfplus.trainer.data import Rollout

class AgentController(ABC):

    def __init__(
            self,  
            state_dim: int,
            action_dim: int 
    ):
        self._state_dim = state_dim 
        self._action_dim = action_dim 


    @property
    def action_dim(self) -> int:
        return self._action_dim
    
    @property
    def state_dim(self) -> int:
        return self._state_dim


    @abstractproperty
    def config(self) -> dict:
        pass

    @abstractproperty
    def actor_params(self) -> Params:
        pass

    @abstractmethod
    def act(self, params: Optional[Params] = None) -> Action:
        pass

    @abstractmethod
    def step(self,  key: PRNGKey, params: Optional[Params] = None) -> Tuple[Action, Array]:
        pass

    @abstractmethod
    def update(self, rollout: Rollout, step: int) -> dict:
        pass

    @abstractmethod
    def save(self, save_dir: str, step: int):
        pass

    @abstractmethod
    def load(self, load_dir: str, step: int):
        pass