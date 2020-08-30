from abc import ABC
from collections import defaultdict
from typing import List, Dict, Any

import gym
import numpy as np

from racecar_gym.envs.specs import TaskSpec


class RewardRange:
    def __init__(self, min: float, max: float):
        self._min, self._max = min, max

    def min(self):
        return self._min

    def max(self):
        return self._max

    def __contains__(self, reward):
        return reward >= self.min() and reward <= self.max()

class Task(ABC):

    def reward_range(self) -> RewardRange:
        raise NotImplementedError

    def reward(self, state, action) -> float:
        raise NotImplementedError()

    def done(self, state) -> bool:
        raise NotImplementedError()

class TimeBasedRacingTask(Task):

    def __init__(self, max_time: float, time_step: float, laps: int):
        self._max_time = max_time
        self._time_step = time_step
        self._laps = laps

    def reward_range(self) -> RewardRange:
        return RewardRange(-self._max_time, 0)

    def reward(self, state, action) -> float:
        if state['collision']:
            return -self._max_time
        return -self._time_step


    def done(self, state) -> bool:
        return state['time'] > self._max_time\
               or state['collision']\
               or state['lap'] > self._laps

def from_spec(spec: TaskSpec):
    if spec.task_name == 'time_based':
        return TimeBasedRacingTask(**spec.params)


class MultiAgentTask:
    def __init__(self, assigned_tasks: Dict[str, Task]):
        self._tasks = assigned_tasks

    def reward(self, states: Dict, actions: Dict) -> Dict[Any, float]:
        rewards = defaultdict(float)
        for agent in self._tasks.keys():
            rewards[agent] += self._tasks[agent].reward(states[agent], actions[agent])
        return rewards

    def done(self, states: Dict):
        return all([task.done(states[agent]) for agent, task in self._tasks.items()])
