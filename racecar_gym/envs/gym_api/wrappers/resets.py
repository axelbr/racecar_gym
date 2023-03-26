from typing import Any, Optional, Dict, Tuple

import gymnasium
from gymnasium.core import WrapperObsType, ObsType


class FixedResetMode(gymnasium.Wrapper):

    def __init__(self, env, mode: str):
        super().__init__(env)
        self._mode = mode

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[ObsType, Dict[str, Any]]:
        options = options or {}
        return self.env.reset(seed=seed, options={**options, 'mode': self._mode})

