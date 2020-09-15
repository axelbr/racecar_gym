from typing import Generic, List

import gym

from racecar_gym.models.definitions import Observation, Action, StepReturn
from racecar_gym.models.tasks import Task
from racecar_gym.models.vehicles import Vehicle


class Agent(Generic[Observation, Action]):

    def __init__(self, vehicle: Vehicle, task: Task, sensors: List[str], time_step: float):
        self._vehicle = vehicle
        self._task = task
        self._available_sensors = sensors
        self._time = 0.0
        self._time_step = time_step

    @property
    def action_space(self) -> gym.Space:
        return self._vehicle.action_space

    @property
    def observation_space(self) -> gym.Space:
        return self._vehicle.observation_space

    def step(self, action: Action) -> StepReturn:
        self._time += self._time_step
        observation, info = self._vehicle.query(sensors=self._available_sensors)
        self._vehicle.control(dict(velocity=action[0], steering_angle=action[1], force=action[2]))
        observation['time'] = self._time
        done = self._task.done(observation)
        reward = self._task.reward(observation, action)
        return observation, reward, done, info

    def reset(self) -> Observation:
        self._vehicle.reset()
        self._time = 0
        observation, info = self._vehicle.query(sensors=self._available_sensors)
        return observation