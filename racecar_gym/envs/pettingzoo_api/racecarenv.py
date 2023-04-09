import functools
from typing import Dict, Optional, Tuple, List

import gymnasium
import numpy as np
from pettingzoo import ParallelEnv
from pettingzoo.utils.env import ObsDict, ActionDict, AgentID

from ..gym_api import MultiAgentRaceEnv


class _MultiAgentRaceEnv(ParallelEnv):

    metadata = {'render.modes': ['follow', 'birds_eye'], "name": "racecar_v1"}

    def __init__(self, scenario: str, render_mode: str = 'follow', render_options: Dict = None):
        self._env = MultiAgentRaceEnv(
            scenario=scenario,
            render_mode=render_mode,
            render_options=render_options,
        )
        self.agents = self._env.scenario.agents.keys()
        self.action_spaces = self._env.action_space # type: ignore
        self.observation_spaces = self._env.observation_space

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return self._env.observation_space.spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return self._env.action_space.spaces[agent]

    def reset(self, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None) -> ObsDict:
        obs, info = self._env.reset(seed=seed, options=options)
        if return_info:
            return obs, info
        else:
            return obs

    def step(self, actions: ActionDict) -> Tuple[
        ObsDict, Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, dict]
    ]:
        return self._env.step(actions) # type: ignore


    def render(self) -> None | np.ndarray | str | List:
        return self._env.render()

