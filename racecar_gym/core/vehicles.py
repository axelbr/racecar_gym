import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import gymnasium

from .actuators import Actuator
from .definitions import Pose
from .sensors import Sensor


class Vehicle(ABC):

    def control(self, commands: Dict) -> None:
        for actuator, command in commands.items():
            self.actuators[actuator].control(command)

    def observe(self) -> Dict[str, Any]:
        observations = {}
        for sensor in self.sensors:
            observations[sensor.name] = sensor.observe()
        return observations

    @property
    @abstractmethod
    def id(self) -> Any:
        pass

    @property
    @abstractmethod
    def sensors(self) -> List[Sensor]:
        pass

    @property
    @abstractmethod
    def actuators(self) -> Dict[str, Actuator]:
        pass

    @property
    def action_space(self) -> gymnasium.spaces.Dict:
        return gymnasium.spaces.Dict(dict((name, actuator.space()) for name, actuator in self.actuators.items()))

    @property
    def observation_space(self) -> gymnasium.spaces.Dict:
        spaces = dict((s.name, s.space()) for s in self.sensors)
        return gymnasium.spaces.Dict(spaces)

    @abstractmethod
    def reset(self, pose: Pose):
        pass
