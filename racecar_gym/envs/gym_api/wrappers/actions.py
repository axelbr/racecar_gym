import gymnasium
import numpy as np

from racecar_gym.envs.gym_api import MultiAgentRaceEnv


class FlattenActionWrapper(gymnasium.ActionWrapper):
    """
    Flattens the nested action dictionary to numpy arrays. Only Box spaces are supported.
    """

    def __init__(self, env: MultiAgentRaceEnv):
        super().__init__(env)
        self._forward_mappings = dict((key, i) for i, key in enumerate(env.action_space.spaces.keys()))
        self._reverse_mappings = dict((i, key) for i, key in enumerate(env.action_space.spaces.keys()))
        self._build_spaces(base_env=env)

    def _build_spaces(self, base_env: gymnasium.Env):
        new_action_space = dict()
        for agent, action_space in base_env.action_space.spaces.items():
            low = np.concatenate([space.low for space in action_space.spaces.values()])
            high = np.concatenate([space.high for space in action_space.spaces.values()])
            new_action_space[agent] = gymnasium.spaces.Box(low, high)
        self.action_space = gymnasium.spaces.Dict(new_action_space)

    def reverse_action(self, action):
        for agent in action:
            actions = []
            mappings = sorted(self.env.action_space.spaces[agent].spaces.keys())
            for key in mappings:
                actions.append(action[agent][key])
            action[agent] = np.concatenate(actions, axis=0)
        return action

    def action(self, action):
        action_dict = dict()
        for agent in action:
            action_dict[agent] = dict()
            mappings = sorted(self.env.action_space.spaces[agent].spaces.keys())
            for idx, key in enumerate(mappings):
                action_dict[agent][key] = action[agent][idx]
        return action_dict


# class FlattenActionWrapper(gym.ActionWrapper):
#
#     def __init__(self, env):
#         assert isinstance(env.action_space, gym.spaces.Dict)
#         assert all(isinstance(space, gym.spaces.Box) for space in env.action_space.spaces.values())
#         super().__init__(env)
#         self.action_space = gym.spaces.Box(
#             low=np.array([env.action_space.spaces[k].low for k in sorted(env.action_space.spaces.keys())]),
#             high=np.array([env.action_space.spaces[k].low for k in sorted(env.action_space.spaces.keys())])
#         )
#
#     def action(self, action):
#         action = np.array([action[k] for k in sorted(action.keys())])
#         return action
#
#     def reverse_action(self, action):
#         actions = dict()
#         offset = 0
#         for k in sorted(self.env.action_space.spaces.keys()):
#             shape = self.env.action_space.spaces[k].shape
#             actions[k] = action[offset:shape[0]]
#             offset += shape[0]
#         return actions