from dataclasses import dataclass
from typing import Tuple

import gym
import numpy as np
import pybullet

from racecar_gym.entities import actuators


class Motor(actuators.Motor):
    @dataclass
    class Config:
        body_id: int
        link_index: int
        velocity_multiplier: float
        max_velocity: float
        max_force: float

    def __init__(self, name: str, config: Config):
        super().__init__(name)
        self._config = config

    def control(self, command: Tuple[float, float]) -> None:
        velocity, force = command
        pybullet.setJointMotorControl2(self._config.body_id,
                                       self._config.link_index, pybullet.VELOCITY_CONTROL,
                                       targetVelocity=velocity * self._config.velocity_multiplier,
                                       force=force)

    def space(self) -> gym.Space:
        return gym.spaces.Box(low=np.array([-self._config.max_velocity, 0.0]),
                              high=np.array([self._config.max_velocity, self._config.max_force]),
                              shape=(2,))


class SteeringWheel(actuators.SteeringWheel):
    @dataclass
    class Config:
        body_id: int
        link_index: int
        steering_multiplier: float
        max_steering_angle: float

    def __init__(self, name: str, config: Config):
        super().__init__(name)
        self._config = config

    def control(self, command: float) -> None:
        pybullet.setJointMotorControl2(self._config.body_id,
                                       self._config.link_index,
                                       pybullet.POSITION_CONTROL,
                                       targetPosition=-command * self._config.steering_multiplier)

    def space(self) -> gym.Space:
        return gym.spaces.Box(low=-self._config.max_steering_angle,
                              high=self._config.max_steering_angle,
                              shape=(1,))
