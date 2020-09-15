from typing import Generic
import gym
from racecar_gym.models.definitions import Observation


class Sensor(Generic[Observation]):

    def space(self) -> gym.Space:
        pass

    def observe(self) -> Observation:
        pass


class FixedTimestepSensor(Sensor[Observation], Generic[Observation]):

    def __init__(self, sensor: Sensor, frequency: int, time_step: float):
        self._sensor = sensor
        self._frequency = 1 / frequency
        self._time_step = time_step
        self._last_timestep = self._frequency
        self._last_observation = None

    def space(self) -> gym.Space:
        return self._sensor.space()

    def observe(self) -> Observation:
        self._last_timestep += self._time_step
        if self._last_timestep >= self._frequency:
            self._last_observation = self._sensor.observe()
            self._last_timestep = 0
        return self._last_observation