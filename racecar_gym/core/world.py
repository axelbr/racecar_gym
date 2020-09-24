from abc import ABC, abstractmethod
from typing import Dict, Any

import gym

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
    def initial_pose(self, position: int) -> Pose:
        pass

    @abstractmethod
    def space(self) -> gym.Space:
        pass

    @abstractmethod
    def state(self, vehicle_id: Any) -> Dict[str, Any]:
        pass


