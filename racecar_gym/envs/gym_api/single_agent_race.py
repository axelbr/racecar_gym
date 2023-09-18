from __future__ import annotations
from typing import Dict, Any, Tuple, Optional, SupportsFloat, Union
import gymnasium
import numpy as np
from gymnasium.core import ObsType, ActType, RenderFrame

from racecar_gym.envs.scenarios import SingleAgentScenario

class SingleAgentRaceEnv(gymnasium.Env):

    metadata = {
        'render_modes': ['human', 'rgb_array_follow', 'rgb_array_birds_eye', 'rgb_array_lidar']
    }

    def __init__(self, scenario: str, render_mode: str = 'human', render_options: Optional[Dict[str, Any]] = None):
        scenario = SingleAgentScenario.from_spec(scenario, rendering=render_mode == 'human')
        self._scenario = scenario
        self._initialized = False
        self._render_mode = render_mode
        self._render_options = render_options or {}
        self.action_space = scenario.agent.action_space

    @property
    def observation_space(self):
        space = self._scenario.agent.observation_space
        space.spaces['time'] = gymnasium.spaces.Box(low=0, high=1, shape=())
        return space

    @property
    def scenario(self):
        return self._scenario

    def step(self, action: ActType) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        assert self._initialized, 'Reset before calling step'
        observation, info = self._scenario.agent.step(action=action)
        self._scenario.world.update()
        state = self._scenario.world.state()
        observation['time'] = np.array(state[self._scenario.agent.id]['time'], dtype=np.float32)
        done = self._scenario.agent.done(state)
        reward = self._scenario.agent.reward(state, action)
        return observation, reward, done, False, state[self._scenario.agent.id]

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[ObsType, Dict[str, Any]]:
        super().reset(seed=seed, options=options)
        if not self._initialized:
            self._scenario.world.init()
            self._initialized = True
        else:
            self._scenario.world.reset()
        if options is not None and 'mode' in options:
            mode = options['mode']
        else:
            mode = 'grid'
        obs = self._scenario.agent.reset(self._scenario.world.get_starting_position(self._scenario.agent, mode))
        self._scenario.world.update()
        state = self._scenario.world.state()
        obs['time'] = np.array(state[self._scenario.agent.id]['time'], dtype=np.float32)
        return obs, state[self._scenario.agent.id]

    def render(self) -> Union[RenderFrame, list[RenderFrame], None]:
        if self._render_mode == 'human':
            return None
        else:
            mode = self._render_mode.replace('rgb_array_', '')
            return self._scenario.world.render(mode=mode, agent_id=self._scenario.agent.id, **self._render_options)