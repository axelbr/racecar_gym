import gym

class FixedResetMode(gym.Wrapper):

    def __init__(self, env, mode: str):
        super().__init__(env)
        self._mode = mode

    def reset(self):
        return self.env.reset(mode=self._mode)

