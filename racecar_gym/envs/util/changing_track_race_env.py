import random
from typing import Callable, List

import gym

from racecar_gym.envs.util.subprocess_env import SubprocessEnv


class ChangingTrackRaceEnv(gym.Env):

    def __init__(self, env_factories: List[Callable[[], gym.Env]], order: str = 'sequential'):
        super().__init__()
        self._current_track_index = 0
        if order == 'sequential':
            self._order_fn = lambda: (self._current_track_index + 1) % len(env_factories)
        elif order == 'random':
            self._order_fn = lambda: random.choice(list(set(range(0, len(env_factories))) - {self._current_track_index}))
        elif order == 'manual':
            self._order_fn = lambda: self._current_track_index
        self._order = order


        self._envs = [
            SubprocessEnv(factory=factory, blocking=True)
            for factory
            in env_factories
        ]
        assert all(self._envs[0].action_space == env.action_space for env in self._envs)
        assert all(self._envs[0].observation_space == env.observation_space for env in self._envs)
        self.action_space = self._envs[0].action_space
        self.observation_space = self._envs[0].observation_space


    def step(self, action):
        return self._get_env().step(action=action)

    def reset(self, mode: str = 'grid'):
        self._current_track_index = self._order_fn()
        return self._get_env().reset(mode=mode)

    def render(self, mode='follow', **kwargs):
        return self._get_env().render(mode=mode, **kwargs)

    def close(self):
        for env in self._envs:
            env.close()

    def _get_env(self):
        return self._envs[self._current_track_index]

    def set_next_env(self):
        assert self._order == 'manual'
        self._current_track_index = (self._current_track_index + 1) % len(self._envs)
