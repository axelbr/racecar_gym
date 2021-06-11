import gym
import numpy as np

class MultiAgentStackingWrapper(gym.ObservationWrapper):

    def __init__(self, env: gym.Env, horizon: int):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Dict) and isinstance(env.action_space, gym.spaces.Dict)
        self._horizon = horizon
        self._step = 0
        self._history = {}
        self._build_spaces(base_env=env, horizon=horizon)
        self._history = self.observation_space.sample()

    def _build_spaces(self, base_env: gym.Env, horizon: int):
        new_obs_space = dict()
        for agent in base_env.observation_space.spaces:
            new_obs_space[agent] = gym.spaces.Dict()
            obs_space = base_env.observation_space.spaces[agent]
            for obs_key, space in obs_space.spaces.items():
                if isinstance(space, gym.spaces.Box):
                    low = np.repeat([space.low], horizon, axis=0)
                    high = np.repeat([space.high], horizon, axis=0)
                    new_obs_space[agent].spaces[obs_key] = gym.spaces.Box(low, high)
                else:
                    raise NotImplementedError('No other spaces are yet implemented for stacking!')

        self.observation_space = gym.spaces.Dict(new_obs_space)

    def _reset_history(self):
        sample = self.observation_space.sample()
        self._history = dict()
        for agent, observation_dict in sample.items():
            self._history[agent] = dict()
            for key, observation in observation_dict.items():
                self._history[agent][key] = np.zeros_like(observation)

    def reset(self, **kwargs):
        self._reset_history()
        return super().reset(**kwargs)

    def observation(self, observation):
        for agent, obs_dict in observation.items():
            for obs_key, obs in obs_dict.items():
                history = self._history[agent][obs_key]
                self._history[agent][obs_key] = np.concatenate([history[1:], [obs]], axis=0)
        return self._history
