from typing import List, Tuple, Dict

from gym import Env

from .scenarios import MultiAgentScenario
from .multi_agent_race import MultiAgentRaceEnv
from racecar_gym.envs.util.vectorized_race import VectorizedRaceEnv


class VectorizedMultiAgentRaceEnv(Env):

    metadata = {'render.modes': ['follow', 'birds_eye', 'lidar']}

    def __init__(self, scenarios: List[MultiAgentScenario]):
        self._env = VectorizedRaceEnv(factories=[lambda: MultiAgentRaceEnv(s) for s in scenarios])
        self.action_space, self.observation_space = self._env.action_space, self._env.observation_space


    def step(self, actions: Tuple[Dict]):
        return self._env.step(actions=actions)

    def reset(self, mode: str = 'grid'):
        return self._env.reset(mode=mode)

    def close(self):
        self._env.close()

    def render(self, mode='follow', agent: str = None, **kwargs):
        return self._env.render(mode=mode, agent=agent, **kwargs)

