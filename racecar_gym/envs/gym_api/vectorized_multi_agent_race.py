from typing import List, Tuple, Dict, Any, Optional

import gymnasium
from gymnasium.core import ObsType

from . import MultiAgentRaceEnv
from .vectorized_race import VectorizedRaceEnv


class VectorizedMultiAgentRaceEnv(gymnasium.Env):
    metadata = {
        'render_modes': ['human', 'rgb_array_follow', 'rgb_array_birds_eye', 'rgb_array_lidar']
    }

    def __init__(self, scenarios: List[str], render_mode: str = 'human', render_options: Dict = None):
        self._env = VectorizedRaceEnv(
            factories=[
                lambda: MultiAgentRaceEnv(s, render_mode=render_mode, render_options=render_options)
                for s
                in scenarios
            ]
        )
        self.action_space, self.observation_space = self._env.action_space, self._env.observation_space

    def step(self, actions: Tuple[Dict]):
        return self._env.step(actions=actions)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[ObsType, Dict[str, Any]]:
        return self._env.reset(seed=seed, options=options)

    def close(self):
        self._env.close()

    def render(self):
        return self._env.render()

