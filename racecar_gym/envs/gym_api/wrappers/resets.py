from typing import Any

import gymnasium
from gymnasium.core import WrapperObsType


class FixedResetMode(gymnasium.Wrapper):

    def __init__(self, env, mode: str):
        super().__init__(env)
        self._mode = mode

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[
        WrapperObsType, dict[str, Any]]:
        options = options or {}
        return self.env.reset(seed=seed, options={**options, 'mode': self._mode})

