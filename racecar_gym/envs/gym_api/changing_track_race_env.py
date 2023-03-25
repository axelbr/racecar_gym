import random
from typing import Callable, List, Any, Optional, Dict, Tuple

import gymnasium
from gymnasium.core import ObsType

from .subprocess_env import SubprocessEnv


class ChangingTrackRaceEnv(gymnasium.Env):

    metadata = {
        'render_modes': ['human', 'rgb_array_follow', 'rgb_array_birds_eye', 'rgb_array_lidar']
    }

    def __init__(self, env_factories: List[Callable[[], gymnasium.Env]], order: str = 'sequential', render_mode: str = 'follow'):
        super().__init__()
        self._current_track_index = 0
        self._render_mode = render_mode
        if order == 'sequential':
            self._order_fn = lambda: (self._current_track_index + 1) % len(env_factories)
        elif order == 'random':
            self._order_fn = lambda: random.choice(list(set(range(0, len(env_factories))) - {self._current_track_index}))
        elif order == 'manual':
            self._order_fn = lambda: self._current_track_index
        self._order = order


        self._envs = [
            SubprocessEnv(factory=factory, blocking=True)
            for factory
            in env_factories
        ]
        assert all(self._envs[0].action_space == env.action_space for env in self._envs)
        assert all(self._envs[0].observation_space == env.observation_space for env in self._envs)
        self.action_space = self._envs[0].action_space
        self.observation_space = self._envs[0].observation_space


    def step(self, action):
        return self._get_env().step(action=action)

    def reset(self, *, seed: Optional[int] = None, options: Dict[str, Any] = None) -> Tuple[ObsType, Dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self._current_track_index = self._order_fn()
        if options is not None:
            mode = options.get('mode', 'grid')
        else:
            mode = 'grid'
        options = options or {}
        return self._get_env().reset(seed=seed, options={'mode': mode, **options})

    def render(self):
        return self._get_env().render()

    def close(self):
        for env in self._envs:
            env.close()

    def _get_env(self):
        return self._envs[self._current_track_index]

    def set_next_env(self):
        assert self._order == 'manual'
        self._current_track_index = (self._current_track_index + 1) % len(self._envs)
