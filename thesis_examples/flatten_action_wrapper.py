import numpy as np

import gym
from gym import ActionWrapper
from gym.spaces import Box
from gymnasium import spaces

class FlattenAction(ActionWrapper):

    def __init__(self, env: gym.Env):
        """A wrapper for clipping continuous actions within the valid bound.
        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)
        self.action_space = spaces.flatten_space(env.action_space)

    def action(self, action):
        """Clips the action within the valid bounds.
        Args:
            action: The action to clip
        Returns:
            The clipped action
        """
        return spaces.flatten(self.env.observation_space, action)