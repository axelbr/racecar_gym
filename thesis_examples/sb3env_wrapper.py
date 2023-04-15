

# A wrapper class for training racecar env in stable baselines
from __future__ import annotations
import functools
from typing import Dict, Optional, Tuple, List

import gymnasium

#from ..gym_api import MultiAgentRaceEnv

from racecar_gym.envs.gym_api import MultiAgentRaceEnv

import numpy as np

from dictionary_space_utility import unwrap_obs_space, flatten_obs_space,flatten_acts_space



class SB3Wrapper(gymnasium.Env):

    def __init__(self, env):


        self.agents = list(env.scenario.agents.keys())
        self.possible_agents = list(env.scenario.agents.keys())
        self.observation_spaces = env.observation_space
        self.action_spaces = env.action_space

        self.render_mode = env.render_mode

        self._env = env

        self.observation_space = self.observation_state_c()

        self.action_space = self.action_space('A')
        #print("action",self.action_space)

        # adding render_mode to alleviate an error with sb3


    #@property
    def observation_state_c(self):

        keys = self._env.observation_space.keys()
        keys = list(keys)

        #unwrapping the original observation state such that it is no longer a nested dictionary
        new_obs = unwrap_obs_space(self._env)

        #new obs_state to be used with sb3
        obs_state = new_obs[keys[0]]

        #obs_state = gymnasium.spaces.flatten_space(obs_state)
        #print("flattened", obs_state)

        #return obs_state
        return flatten_obs_space()

    def action_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        act = self._env.action_space.spaces[agent]

        #flattening the action space
        #action_space = gymnasium.spaces.flatten_space(act)
        #action_space = flatten_acts_space(act)

        action_space = flatten_acts_space()
        return action_space


    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> ObsDict:
        obs, _ = self._env.reset(seed=seed, options=options)
        return obs

    #not sure if these functions will work work with the new observation step
    def step(self, actions: ActionDict) -> Tuple[
        ObsDict, Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, dict]
    ]:
        return self._env.step(actions) # type: ignore


    def render(self) -> None | np.ndarray | str | List:
        return self._env.render()

    #messing around with trying to flatten action spaces

    def flatten(self):
        return self._env.flatten()









