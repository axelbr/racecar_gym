from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import gym

T = TypeVar('T')


class Sensor(Generic[T], ABC):

    def __init__(self, name: str):
        self._name = name

    @abstractmethod
    def space(self) -> gym.Space:
        pass

    @abstractmethod
    def observe(self) -> T:
        pass

    @property
    def name(self):
        return self._name
