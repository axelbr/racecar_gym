from abc import ABC, abstractmethod
from typing import Dict, Any

import gymnasium
import numpy as np

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
    def get_starting_position(self, agent: Agent, mode: str) -> Pose:
        pass

    @abstractmethod
    def space(self) -> gymnasium.Space:
        pass

    @abstractmethod
    def state(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def render(self, agent_id: str, mode: str, width: int = 640, height: int = 480) -> np.ndarray:
        pass

    @abstractmethod
    def seed(self, seed: int = None):
        pass

