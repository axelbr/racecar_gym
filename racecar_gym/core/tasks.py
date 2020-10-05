import math
from abc import ABC, abstractmethod
from typing import List

import numpy as np

from .specs import TaskSpec


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

    @abstractmethod
    def reward_range(self) -> RewardRange:
        pass

    @abstractmethod
    def reward(self, agent_id, state, action) -> float:
        pass

    @abstractmethod
    def done(self, agent_id, state) -> bool:
        pass


class MultiTask(Task):
    def __init__(self, tasks: List[Task], all_done: bool = True, weights: List[float] = None):
        self._tasks = tasks
        self._all_done = all_done
        if weights:
            assert len(weights) == len(tasks)
            self._weights = weights

    def reward_range(self) -> RewardRange:
        return RewardRange(-np.inf, np.inf)

    def reward(self, agent_id, state, action) -> float:
        if self._weights:
            return sum([t.reward(agent_id, state, action) * w for t, w in zip(self._tasks, self._weights)])
        else:
            return sum([t.reward(agent_id, state, action) for t in self._tasks])

    def done(self, agent_id, state) -> bool:
        if self._all_done:
            return all([t.done(agent_id, state) for t in self._tasks])
        else:
            return any([t.done(agent_id, state) for t in self._tasks])


class TimeBasedRacingTask(Task):

    def __init__(self, max_time: float):
        self._max_time = max_time
        self._last_section = 0

    def reward_range(self) -> RewardRange:
        return RewardRange(-math.inf, math.inf)

    def reward(self, agent_id, state, action) -> float:
        reward = 0
        section = state[agent_id]['section']
        if section > self._last_section:
            reward += 1.0
            self._last_section = section
        if state[agent_id]['collision']:
            reward -= 1.0
        return reward

    def done(self, agent_id, state) -> bool:
        return state[agent_id]['time'] > self._max_time

class RankDiscountedProgressTask(Task):

    def __init__(self, laps: int):
        self._last_section = -1
        self._current_lap = 0
        self._laps = laps
        self.leading = []

    def reward_range(self) -> RewardRange:
        pass

    def reward(self, agent_id, state, action) -> float:
        rank = 1
        for id in state:
            equal_laps =  state[id]['lap'] == state[agent_id]['lap']
            equal_sections =  state[id]['section'] == state[agent_id]['section']
            if state[id]['lap'] > state[agent_id]['lap']:
                rank += 1
            elif equal_laps and state[id]['section'] > state[agent_id]['section']:
                rank += 1
            elif equal_laps and equal_sections and state[id]['section_time'] < state[agent_id]['section_time']:
                rank += 1

        reward = 0
        section = state[agent_id]['section']

        if self._current_lap < state[agent_id]['lap']:
            self._last_section = -1
            self._current_lap = state[agent_id]['lap']

        if section > self._last_section:
            reward += 1.0 / float(rank)
            self._last_section = section
        if state[agent_id]['collision']:
            reward -= 1.0

        return reward


    def done(self, agent_id, state) -> bool:
        return state[agent_id]['lap'] > self._laps


tasks = {
    'time_based': TimeBasedRacingTask,
    'rank_discounted': RankDiscountedProgressTask
}


def task_from_spec(spec: TaskSpec) -> Task:
    if spec.task_name in tasks:
        return tasks[spec.task_name](**spec.params)


def register_task(name: str, task: Task):
    tasks[name] = task