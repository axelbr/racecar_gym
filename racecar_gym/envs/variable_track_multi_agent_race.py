import random
from abc import ABC
from typing import List, Tuple, Dict

import gym
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection

from gym import Env

from .scenarios import MultiAgentScenario
from .multi_agent_race import MultiAgentRaceEnv
from .subprocess_utils import SubprocessEnv
from .vectorized_multi_agent_race import VectorizedMultiAgentRaceEnv


class DynamicTrackMultiAgentRaceEnv(Env):

    def __init__(self, scenarios: List[MultiAgentScenario], order: str = 'sequential'):
        super().__init__()

        self._current_track_index = 0
        if order == 'sequential':
            self._order_fn = lambda: (self._current_track_index + 1) % len(scenarios)
        elif order == 'random':
            self._order_fn = lambda: random.choice(list(set(range(0, len(scenarios))) - {self._current_track_index}))

        self._envs = [
            SubprocessEnv(factory=lambda: MultiAgentRaceEnv(scenario=scenario), blocking=True)
            for scenario
            in scenarios
        ]
        assert all(self._envs[0].action_space == env.action_space for env in self._envs)
        assert all(self._envs[0].observation_space == env.observation_space for env in self._envs)
        self.action_space = self._envs[0].action_space
        self.observation_space = self._envs[0].observation_space


    def step(self, action: Dict):
        return self._get_env().step(action=action)

    def reset(self, mode: str = 'grid'):
        self._current_track_index = self._order_fn()
        return self._get_env().reset(mode=mode)

    def render(self, mode='follow', agent: str = None, **kwargs):
        return self._get_env().render(mode=mode, agent=agent, **kwargs)

    def _get_env(self):
        return self._envs[self._current_track_index]




