from abc import ABC
from dataclasses import dataclass
from typing import Tuple, TypeVar, List

import gymnasium
import numpy as np
import pybullet

from racecar_gym.core import actuators

T = TypeVar('T')


class BulletActuator(actuators.Actuator[T], ABC):
    def __init__(self, name: str):
        super().__init__(name)
        self._body_id = None
        self._joint_indices = []

    def reset(self, body_id: int, joint_indices: List[int] = None):
        self._body_id = body_id
        self._joint_indices = joint_indices

    @property
    def body_id(self) -> int:
        return self._body_id

    @property
    def joint_indices(self) -> List[int]:
        return self._joint_indices


class Motor(BulletActuator[Tuple[float, float]]):
    @dataclass
    class Config:
        velocity_multiplier: float
        max_velocity: float
        max_force: float

    def __init__(self, name: str, config: Config):
        super().__init__(name)
        self._config = config

    def control(self, acceleration: float) -> None:
        acceleration = np.clip(acceleration, -1, +1)
        if acceleration < 0:
            velocity = 0
        else:
            velocity = self._config.max_velocity * self._config.velocity_multiplier

        force = abs(acceleration) * self._config.max_force

        for index in self.joint_indices:
            pybullet.setJointMotorControl2(
                self.body_id, index,
                pybullet.VELOCITY_CONTROL,
                targetVelocity=velocity,
                force=force
            )

    def space(self) -> gymnasium.Space:
        return gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)


class Speed(BulletActuator[Tuple[float, float]]):
    @dataclass
    class Config:
        velocity_multiplier: float
        max_velocity: float
        max_force: float

    def __init__(self, name: str, config: Config):
        super().__init__(name)
        self._config = config

    def control(self, target_speed: float) -> None:
        """ target_speed is assumed to be mapped from [0,max_velocity] to [-1, +1]"""
        target_speed = np.clip(target_speed, -1, +1)  # sanity check
        target_speed = (target_speed + 1.0) / 2.0 * self._config.max_velocity  # convert to actual range

        velocity = target_speed * self._config.velocity_multiplier
        force = self._config.max_force

        for index in self.joint_indices:
            pybullet.setJointMotorControl2(
                self.body_id, index,
                pybullet.VELOCITY_CONTROL,
                targetVelocity=velocity,
                force=force
            )

    def space(self) -> gymnasium.Space:
        return gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)


class SteeringWheel(BulletActuator[float]):
    @dataclass
    class Config:
        steering_multiplier: float
        max_steering_angle: float

    def __init__(self, name: str, config: Config):
        super().__init__(name)
        self._config = config

    def control(self, command: float) -> None:
        angle = command * self._config.max_steering_angle * self._config.steering_multiplier
        for joint in self.joint_indices:
            pybullet.setJointMotorControl2(
                self.body_id,
                joint,
                pybullet.POSITION_CONTROL,
                targetPosition=-angle
            )

    def space(self) -> gymnasium.Space:
        return gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
