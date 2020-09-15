from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import gym


class Vehicle(ABC):

    @abstractmethod
    def control(self, command) -> None:
        pass

    @property
    @abstractmethod
    def action_space(self) -> gym.Space:
        pass

    @property
    @abstractmethod
    def observation_space(self) -> gym.Space:
        pass

    @abstractmethod
    def query(self, sensors: List[str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        pass

    @abstractmethod
    def reset(self):
        pass