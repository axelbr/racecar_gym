import math
from dataclasses import dataclass

import gym
import numpy as np
import pybullet
from gym import spaces
from pybullet_utils.bullet_client import BulletClient

from racecar_gym.map import Map
from racecar_gym.racecar import RaceCar, Lidar


@dataclass
class SingleCarEnvScenario:
    map: str
    car: str
    initial_pose: np.ndarray

class SingleCarEnv(gym.Env):

    def __init__(self, scenario: SingleCarEnvScenario):
        self._was_reset = False
        self._rendering = None
        self._scenario = scenario
        self.observation_space = spaces.Dict({
            'lidar': spaces.Box(low=np.zeros(100), high=np.zeros(100) + 5),
            'pose': spaces.Box(low=np.array([-10, -10, -10, -math.pi, -math.pi, -math.pi]),
                               high=np.array([10, 10, 10, math.pi, math.pi, math.pi])),
            'velocity': spaces.Box(low=np.array([-10, -10, -10, -math.pi, -math.pi, -math.pi]),
                                   high=np.array([10, 10, 10, math.pi, math.pi, math.pi]))
        })
        self.action_space = spaces.Box(low=np.array([-5, -1]), high=np.array([5, 1]))

    def step(self, action: np.ndarray):
        assert self._was_reset, 'Please reset before interacting with the environment!'
        self._car.step(velocity=action[0], steering_angle=action[1], force=20)
        self._client.stepSimulation()
        observation = self._car.observe(sensors=['odometry', 'lidar'])
        return observation, 0.0, False, {}

    def reset(self):
        if self._rendering:
            self._client = BulletClient(connection_mode=pybullet.GUI)
        else:
            self._client = BulletClient()

        self._client.resetSimulation()
        self._client.setGravity(0, 0, -9.81)

        self._car = RaceCar(client=self._client, initial_pose=self._scenario.initial_pose, model=self._scenario.car)
        self._map = Map(client=self._client, model=self._scenario.map)

        self._was_reset = True
        return self._car.observe(sensors=['odometry', 'lidar'])

    def render(self, mode='human'):
        self._rendering = mode

    def close(self):
        pass