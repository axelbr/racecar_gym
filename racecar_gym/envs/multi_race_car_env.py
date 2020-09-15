from typing import List, Dict

import gym

from racecar_gym.racing.agent import Agent
from racecar_gym.racing.world import World


class MultiRaceCarEnv(gym.Env):

    def __init__(self, world: World, agents: List[Agent]):
        self._agents = agents
        self._world = world
        self._initialized = False

    def step(self, action: List[Dict]):

        assert self._initialized, 'Reset before calling step'

        observations = []
        dones = []
        rewards = []
        infos = []

        for i, agent in enumerate(self._agents):
            observation, reward, done, info = agent.step(action=action[i])
            observations.append(observation)
            dones.append(done)
            rewards.append(reward)
            infos.append(info)

        return observations, rewards, dones, infos

    def reset(self):
        if not self._initialized:
            self._world.init()
            self._initialized = True
        else:
            self._world.reset()
        return [agent.reset(self._world.initial_pose(i)) for i, agent in enumerate(self._agents)]
