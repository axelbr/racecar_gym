import random
from abc import ABC
from typing import List, Tuple, Dict

import gym
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from .scenarios import MultiAgentScenario
from .multi_agent_race import MultiAgentRaceEnv
from .vectorized_multi_agent_race import VectorizedMultiAgentRaceEnv


class DynamicTrackMultiAgentRaceEnv(VectorizedMultiAgentRaceEnv):

    def __init__(self, scenarios: List[MultiAgentScenario], order: str = 'sequential'):
        super().__init__(scenarios)
        assert all([self.observation_space.spaces[0] == space for space in self.observation_space.spaces])
        assert all([self.action_space.spaces[0] == space for space in self.action_space.spaces])
        self._current_track_index = 0
        if order == 'sequential':
            self._order_fn = lambda: (self._current_track_index + 1) % len(scenarios)
        elif order == 'random':
            self._order_fn = lambda: random.choice(set(range(0, len(scenarios))) - {self._current_track_index})
        self.observation_space = self.observation_space.spaces[0]
        self.action_space = self.action_space.spaces[0]

    def step(self, action: Dict):
        connection = self._env_connections[self._current_track_index]
        connection.send('step')
        connection.send(action)
        obs, reward, done, state = connection.recv()
        return obs, reward, done, state

    def reset(self, mode: str = 'grid'):
        next_track = self._order_fn()
        connection = self._env_connections[next_track]
        connection.send('reset')
        connection.send(mode)
        obs = connection.recv()
        self._current_track_index = next_track
        return obs

    def render(self, mode='follow', agent: str = None, **kwargs):
        connection = self._env_connections[self._current_track_index]
        connection.send('render')
        connection.send((mode, agent, kwargs))
        image = connection.recv()
        return image




