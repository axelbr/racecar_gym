from typing import Any

import gymnasium

from .definitions import Pose, Velocity
from .vehicles import Vehicle
from racecar_gym.tasks import Task


class Agent:

    def __init__(self, id: str, vehicle: Vehicle, task: Task):
        self._id = id
        self._vehicle = vehicle
        self._task = task

    @property
    def vehicle(self):
        return self._vehicle

    @property
    def task(self):
        return self._task

    @property
    def id(self) -> str:
        return self._id

    @property
    def vehicle_id(self) -> Any:
        return self._vehicle.id

    @property
    def action_space(self) -> gymnasium.Space:
        return self._vehicle.action_space

    @property
    def observation_space(self) -> gymnasium.Space:
        return self._vehicle.observation_space

    def step(self, action):
        observation = self._vehicle.observe()
        self._vehicle.control(action)
        return observation, {}

    def done(self, state) -> bool:
        return self._task.done(agent_id=self._id, state=state)

    def reward(self, state, action) -> float:
        return self._task.reward(agent_id=self._id, state=state, action=action)

    def reset(self, pose: Pose):
        self._vehicle.reset(pose=pose)
        self._task.reset()
        observation = self._vehicle.observe()
        return observation
