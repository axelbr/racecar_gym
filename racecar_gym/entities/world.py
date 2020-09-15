from abc import ABC, abstractmethod

from racecar_gym.entities.definitions import Pose


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
