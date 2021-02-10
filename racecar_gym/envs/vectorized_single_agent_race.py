from typing import List, Tuple, Dict

import gym
from .scenarios import SingleAgentScenario
from .single_agent_race import SingleAgentRaceEnv
from racecar_gym.envs.util.vectorized_race import VectorizedRaceEnv


class VectorizedSingleAgentRaceEnv(gym.Env):

    metadata = {'render.modes': ['follow', 'birds_eye', 'lidar']}

    def __init__(self, scenarios: List[SingleAgentScenario]):
        self._env = VectorizedRaceEnv(factories=[lambda: SingleAgentRaceEnv(s) for s in scenarios])
        self.action_space, self.observation_space = self._env.action_space, self._env.observation_space

    def step(self, actions: Tuple[Dict]):
        return self._env.step(actions=actions)

    def reset(self, mode: str = 'grid'):
        return self._env.reset(mode=mode)

    def close(self):
        self._env.close()

    def render(self, mode='follow', **kwargs):
        return self._env.render(mode=mode, **kwargs)

