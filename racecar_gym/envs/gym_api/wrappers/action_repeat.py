import gymnasium
from typing import Callable, List, Any, Dict

import numpy as np


class ActionRepeat(gymnasium.Wrapper):

    def __init__(self, env, steps: int, reward_aggregate_fn: Callable[[List], Any],
                 termination_fn: Callable[[Any, Any], bool]):
        super().__init__(env)
        self._reward_aggregate_fn = reward_aggregate_fn
        self._termination_fn = termination_fn
        self._steps = steps

    def step(self, action):
        obs, info, terminated, truncated = None, None, None, None
        rewards = []
        for i in range(self._steps):
            obs, reward, terminated, truncated, info = self.env.step(action)
            rewards.append(reward)
            if self._termination_fn(terminated, truncated):
                break
        reward = self._reward_aggregate_fn(rewards)
        return obs, reward, terminated, truncated, info

def _aggregate_dicts(dicts, initial_value: Any, agg_fn: Callable[[Any, Any], Any]):
    result = dict((key, initial_value) for key in dicts[0].keys())
    for item in dicts:
        for k, v in item.items():
            result[k] = agg_fn(result[k], v)
    return result


def MultiAgentActionRepeat(env, steps: int):

    def aggregate(rewards):
        return _aggregate_dicts(dicts=rewards, initial_value=0.0, agg_fn=float.__add__)

    def termination(terminated, truncated):
        return any(terminated.values()) or any(truncated.values())

    return ActionRepeat(env=env, steps=steps, reward_aggregate_fn=aggregate, termination_fn=termination)

def SingleAgentActionRepeat(env, steps: int):
    return ActionRepeat(
        env=env,
        steps=steps,
        reward_aggregate_fn=sum,
        termination_fn=lambda terminated, truncated: terminated or truncated,
    )

def VectorizedSingleAgentActionRepeat(env, steps: int):

    def aggregate(rewards):
        aggregated_rewards = np.array(rewards)
        return aggregated_rewards.sum(axis=0)

    def termination(terminated, truncated):
        return all(np.logical_or(terminated, truncated))

    return ActionRepeat(env=env, steps=steps, reward_aggregate_fn=aggregate, termination_fn=termination)
