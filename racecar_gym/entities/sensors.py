from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Any

import gym
import numpy as np
from nptyping import NDArray

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


class FixedTimestepSensor(Sensor[T]):

    def __init__(self, sensor: Sensor, frequency: int, time_step: float):
        self._sensor = sensor
        self._frequency = 1 / frequency
        self._time_step = time_step
        self._last_timestep = self._frequency
        self._last_observation = None

    def space(self) -> gym.Space:
        return self._sensor.space()

    def observe(self) -> T:
        self._last_timestep += self._time_step
        if self._last_timestep >= self._frequency:
            self._last_observation = self._sensor.observe()
            self._last_timestep = 0
        return self._last_observation


class Lidar(Sensor[NDArray[(Any,), np.float]], ABC):
    pass


class RGBCamera(Sensor[NDArray[(Any, Any, 3), np.int]], ABC):
    pass


class IMU(Sensor[NDArray[(6,), np.float]], ABC):
    pass


class Tachometer(Sensor[NDArray[(6,), np.float]], ABC):
    pass


class GPS(Sensor[NDArray[(6,), np.float]], ABC):
    pass


class CollisionSensor(Sensor[bool], ABC):
    pass


class LapCounter(Sensor[int], ABC):
    pass
