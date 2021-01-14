from typing import List, Callable

import gym
from gym import Env

from racecar_gym.envs.util.subprocess_env import SubprocessEnv

class VectorizedRaceEnv(gym.Env):

    metadata = {'render.modes': ['follow', 'birds_eye']}

    def __init__(self, factories: List[Callable[[], Env]]):
        self._envs = [
            SubprocessEnv(factory=factory, blocking=False)
            for factory
            in factories
        ]
        self.observation_space = gym.spaces.Tuple([env.observation_space for env in self._envs])
        self.action_space = gym.spaces.Tuple([env.action_space for env in self._envs])

    def step(self, actions):
        promises = []
        for action, env in zip(actions, self._envs):
            promises.append(env.step(action))

        observations, rewards, dones, states = [], [], [], []
        for promise in promises:
            obs, reward, done, state = promise()
            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            states.append(state)

        return observations, rewards, dones, states

    def reset(self, **kwargs):
        observations = []
        for env in self._envs:
            obs = env.reset(**kwargs)
            observations.append(obs())
        return observations

    def close(self):
        for env in self._envs:
            env.close()

    def render(self, mode: str = 'follow', **kwargs):
        renderings, promises = [], []
        for env in self._envs:
            promises.append(env.render(mode=mode, **kwargs))

        for promise in promises:
            renderings.append(promise())

        return renderings

