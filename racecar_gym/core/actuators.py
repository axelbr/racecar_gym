from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import gymnasium

T = TypeVar('T')

class Actuator(ABC, Generic[T]):

    def __init__(self, name: str):
        self._name = name

    @abstractmethod
    def control(self, command: T) -> None:
        pass

    @abstractmethod
    def space(self) -> gymnasium.Space:
        pass

    @property
    def name(self):
        return self._name