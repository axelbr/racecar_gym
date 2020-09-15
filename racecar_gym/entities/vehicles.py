from abc import ABC, abstractmethod
from typing import Any, Dict, List

import gym

from racecar_gym.entities.actuators import Actuator
from racecar_gym.entities.definitions import Pose
from racecar_gym.entities.sensors import Sensor


class Vehicle(ABC):

    def __init__(self, sensors: List[Sensor], actuators: List[Actuator]):
        self._sensors = sensors
        self._actuators = actuators

    def control(self, commands: Dict) -> None:
        for actuator, command in commands.items():
            self._actuators[actuator].control(command)

    def observe(self) -> Dict[str, Any]:
        observations = {}
        for sensor in self._sensors:
            observations[sensor.name] = sensor.observe()
        return observations

    @property
    def action_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict(dict((a.name, a.space) for a in self._actuators))

    @property
    def observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict(dict((s.name, s.space) for s in self._sensors))

    @abstractmethod
    def reset(self, pose: Pose):
        pass
