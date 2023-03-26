from typing import Optional, Dict, Any, Tuple

import gymnasium
import numpy as np
from gymnasium.core import ObsType


class MultiAgentStackingWrapper(gymnasium.ObservationWrapper):
    """
    Stack observations over time. Adds a time dimension at axis 0 for each observation in the observation dict.
    The latest observation can be retrieved at the last time index. If not enough observations are available to fill
    the whole horizon, zeros are appended.
    Example: lidar scans with shape (1080,) are stacked to arrays of shape (horizon, 1080). Latest scan: obs[-1, :]
    :except NotImplementedError is thrown when env contains any other space type than Box at the observation level.
    """

    def __init__(self, env: gymnasium.Env, horizon: int):
        super().__init__(env)
        assert isinstance(env.observation_space, gymnasium.spaces.Dict) and isinstance(env.action_space, gymnasium.spaces.Dict)
        self._horizon = horizon
        self._step = 0
        self._history = {}
        self._build_spaces(base_env=env, horizon=horizon)
        self._history = self.observation_space.sample()

    def _build_spaces(self, base_env: gymnasium.Env, horizon: int):
        new_obs_space = dict()
        for agent in base_env.observation_space.spaces:
            new_obs_space[agent] = gymnasium.spaces.Dict()
            obs_space = base_env.observation_space.spaces[agent]
            for obs_key, space in obs_space.spaces.items():
                if isinstance(space, gymnasium.spaces.Box):
                    low = np.repeat([space.low], horizon, axis=0)
                    high = np.repeat([space.high], horizon, axis=0)
                    new_obs_space[agent].spaces[obs_key] = gymnasium.spaces.Box(low, high)
                else:
                    raise NotImplementedError('No other spaces are yet implemented for stacking!')

        self.observation_space = gymnasium.spaces.Dict(new_obs_space)

    def _reset_history(self):
        sample = self.observation_space.sample()
        self._history = dict()
        for agent, observation_dict in sample.items():
            self._history[agent] = dict()
            for key, observation in observation_dict.items():
                self._history[agent][key] = np.zeros_like(observation)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[ObsType, Dict[str, Any]]:
        self._reset_history()
        return self.env.reset(seed=seed, options=options)

    def observation(self, observation):
        for agent, obs_dict in observation.items():
            for obs_key, obs in obs_dict.items():
                history = self._history[agent][obs_key]
                self._history[agent][obs_key] = np.concatenate([history[1:], [obs]], axis=0)
        return self._history
