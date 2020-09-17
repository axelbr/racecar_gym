import gym

from .definitions import Pose
from .tasks import Task
from .vehicles import Vehicle


class Agent:

    def __init__(self, id: str, vehicle: Vehicle, task: Task):
        self._id = id
        self._vehicle = vehicle
        self._task = task

    @property
    def id(self) -> str:
        return self._id

    @property
    def action_space(self) -> gym.Space:
        return self._vehicle.action_space

    @property
    def observation_space(self) -> gym.Space:
        return self._vehicle.observation_space

    def step(self, action):
        observation = self._vehicle.observe()
        self._vehicle.control(action)
        return observation, {}

    def done(self, state) -> bool:
        return self._task.done(state)

    def reward(self, state, action) -> float:
        return self._task.reward(state, action)

    def reset(self, pose: Pose):
        self._vehicle.reset(pose=pose)
        observation = self._vehicle.observe()
        return observation
