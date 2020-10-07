from abc import ABC, abstractmethod
from typing import Dict, Any

import gym

from .agent import Agent
from .definitions import Pose


class World(ABC):

    @abstractmethod
    def init(self) -> None:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def update(self) -> float:
        pass

    @abstractmethod
    def get_starting_position(self, agent: Agent) -> Pose:
        pass

    @abstractmethod
    def space(self) -> gym.Space:
        pass

    @abstractmethod
    def state(self) -> Dict[str, Any]:
        pass


