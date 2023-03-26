from typing import List, Callable, Any, Optional, Dict, Tuple

import gymnasium
from gymnasium import Env
from gymnasium.core import ObsType

from .subprocess_env import SubprocessEnv

class VectorizedRaceEnv(gymnasium.Env):

    metadata = {'render.modes': ['follow', 'birds_eye']}

    def __init__(self, factories: List[Callable[[], Env]]):
        self._envs = [
            SubprocessEnv(factory=factory, blocking=False)
            for factory
            in factories
        ]
        self.observation_space = gymnasium.spaces.Tuple([env.observation_space for env in self._envs])
        self.action_space = gymnasium.spaces.Tuple([env.action_space for env in self._envs])

    def step(self, actions):
        promises = []
        for action, env in zip(actions, self._envs):
            promises.append(env.step(action))

        observations, rewards, terminates, truncates, states = [], [], [], [], []
        for promise in promises:
            obs, reward, terminate, truncate, state = promise()
            observations.append(obs)
            rewards.append(reward)
            terminates.append(terminate)
            truncates.append(truncate)
            states.append(state)

        return observations, rewards, terminates, truncates, states

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[ObsType, Dict[str, Any]]:
        observations = []
        for env in self._envs:
            obs = env.reset(seed=seed, options=options)
            observations.append(obs())
        return observations

    def close(self):
        for env in self._envs:
            env.close()

    def render(self, mode: str = 'follow', **kwargs):
        renderings, promises = [], []
        for env in self._envs:
            promises.append(env.render())

        for promise in promises:
            renderings.append(promise())

        return renderings

