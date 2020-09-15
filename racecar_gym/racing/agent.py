import gym

from racecar_gym.entities.definitions import Pose
from racecar_gym.entities.vehicles import Vehicle
from racecar_gym.racing.tasks import Task


class Agent:

    def __init__(self, id: int, vehicle: Vehicle, task: Task):
        self._id = id
        self._vehicle = vehicle
        self._task = task

    @property
    def id(self) -> int:
        return self._id

    @property
    def action_space(self) -> gym.Space:
        return self._vehicle.action_space

    @property
    def observation_space(self) -> gym.Space:
        return self._vehicle.observation_space

    def step(self, action):
        observation = self._vehicle.observe()
        done = self._task.done(observation)
        reward = self._task.reward(observation, action)
        self._vehicle.control(action)
        return observation, reward, done, {}

    def reset(self, pose: Pose):
        self._vehicle.reset(pose=pose)
        observation = self._vehicle.observe()
        return observation
