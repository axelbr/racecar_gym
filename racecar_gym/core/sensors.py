from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import gymnasium

T = TypeVar('T')


class Sensor(Generic[T], ABC):

    def __init__(self, name: str, type: str):
        self._name = name
        self._type = type

    @abstractmethod
    def space(self) -> gymnasium.Space:
        pass

    @abstractmethod
    def observe(self) -> T:
        pass

    @property
    def name(self):
        return self._name

    @property
    def type(self):
        return self._type
