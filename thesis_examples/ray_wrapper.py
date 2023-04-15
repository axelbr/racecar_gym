

# A wrapper class for training racecar env in Ray's RLLIB
from __future__ import annotations
import functools
from typing import Dict, Optional, Tuple, List

import gymnasium

#from ..gym_api import MultiAgentRaceEnv

from racecar_gym.envs.gym_api import MultiAgentRaceEnv

import numpy as np

from dictionary_space_utility import unwrap_obs_space
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from dictionary_space_utility import unwrap_obs_space, flatten_obs_space,flatten_acts_space, flatten_obs, unflatten_acts



class RayWrapper(MultiAgentEnv):

    def __init__(self, env):


        self.agents = list(env.scenario.agents.keys())
        #self.possible_agents = list(env.scenario.agents.keys())
        self.observation_spaces = env.observation_space
        self.action_spaces = env.action_space

        self.render_mode = env.render_mode

        self._env = env

        self.observation_space = self.observation_state_c()

        self.action_space = env.action_space['A']

        # adding render_mode to alleviate an error with sb3

    #@property
    def observation_state_c(self):
        keys = self._env.observation_space.keys()
        keys = list(keys)

        # unwrapping the original observation state such that it is no longer a nested dictionary
        new_obs = unwrap_obs_space(self._env)

        # new obs_state to be used with sb3
        obs_state = new_obs[keys[0]]

        # obs_state = gymnasium.spaces.flatten_space(obs_state)
        # print("flattened", obs_state)

        # return obs_state
        return flatten_obs_space()

    def action_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        act = self._env.action_space.spaces[agent]

        # flattening the action space
        # action_space = gymnasium.spaces.flatten_space(act)
        # action_space = flatten_acts_space(act)

        action_space = flatten_acts_space()
        return action_space

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> ObsDict:
        obs, _ = self._env.reset(seed=seed, options=options)

        new_obs = {}
        for key in self._env.possible_agents:
            new_obs[key] = flatten_obs(obs[key])
            new_obs[key] = new_obs[key].astype(np.float64)
            #print("new_obs type", new_obs[key].dtype, "\n")
            #print(self.observation_space.contains(new_obs[key]))
            #print("shape",self.observation_space.dtype)
            #print(new_obs[key].shape == self.observation_space.shape)
            #print(np.all(new_obs[key] >= self.observation_space.low))
            #print(np.greater_equal(new_obs[key],self.observation_space.low))
            #print(self.observation_space.low)

        return new_obs, {}

    #not sure if these functions will work work with the new observation step
    def step(self, actions: ActionDict) -> Tuple[
        ObsDict, Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, dict]
    ]:
        #unflattening the actions so that multi_agent_race env can process it --> not needed?
        #unflat_acts = unflatten_acts(actions)

        #stepping with the multi_agent_race env's function
        observations, rewards, dones, truncated, state = self._env.step(actions)


        #flattening the observations for rllib
        flat_obs = {}
        for key in self._env.possible_agents:
            flat_obs[key] = flatten_obs(observations[key])
            flat_obs[key] = flat_obs[key].astype(np.float64)

        #return self._env.step(actions) # type: ignore

        #checking if the environment has terminated
        dones.update({'__all__': all(dones.values())})
        truncated.update({'__all__': all(truncated.values())})

        #returning the required values
        return flat_obs, rewards, dones, truncated, state


    def render(self) -> None | np.ndarray | str | List:
        return self._env.render()
