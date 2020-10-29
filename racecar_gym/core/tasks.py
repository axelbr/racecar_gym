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

    @abstractmethod
    def reset(self):
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

    def reset(self):
        self._last_section = 0


class RankDiscountedProgressTask(Task):

    def __init__(self, time_limit: float, laps: int, terminate_on_collision: bool):
        self._last_section = -1
        self._current_lap = 0
        self._laps = laps
        self._time_limit = time_limit
        self._terminate_on_collision = terminate_on_collision
        self.leading = []

    def reward_range(self) -> RewardRange:
        pass

    def reward(self, agent_id, state, action) -> float:
        agents = [(agent, states['lap'], states['progress']) for agent, states in state.items()]
        ranked = [item[0] for item in sorted(agents, key=lambda item: (item[1], item[2]), reverse=True)]
        rank = ranked.index(agent_id) + 1
        section = state[agent_id]['section']
        reward = 0.0
        if not state[agent_id]['collision']:
            if section > self._last_section:
                self._last_section = section
                reward += 1.0 / float(rank)
        else:
            reward -= 5.0
        return reward


    def done(self, agent_id, state) -> bool:
        if self._terminate_on_collision and state[agent_id]['collision']:
            return True

        return state[agent_id]['lap'] > self._laps or self._time_limit < state[agent_id]['time']

    def reset(self):
        self._last_section = -1
        self._current_lap = 0


class ProgressTaskWtPenalty(Task):

    def __init__(self, time_limit: float, laps: int, terminate_on_collision: bool,
                 segment_reward: float = 1000, collision_penalty: float = 100, frame_penalty: float = 0.01):
        self._last_section = -1
        self._current_lap = 0
        self._laps = laps
        self._time_limit = time_limit
        self._terminate_on_collision = terminate_on_collision
        self.leading = []
        self._collision_penalty = collision_penalty
        self._segment_progress_reward = segment_reward
        self._frame_penalty = frame_penalty

    def reward_range(self) -> RewardRange:
        pass

    def reward(self, agent_id, state, action) -> float:
        section = state[agent_id]['section']
        n_segments = state[agent_id]['n_segments']
        reward = 0.0
        if section is not None:
            reward -= self._frame_penalty       # for each new frame, add penalty to learn the agent to be fast
            if self._current_lap < state[agent_id]['lap']:
                self._last_section = -1
                self._current_lap = state[agent_id]['lap']
            if section > self._last_section:
                reward += self._segment_progress_reward / n_segments
                self._last_section = section
            if state[agent_id]['collision']:
                reward -= self._collision_penalty
        return reward


    def done(self, agent_id, state) -> bool:
        if self._terminate_on_collision and state[agent_id]['collision']:
            return True
        if state[agent_id]['lap'] > self._laps or self._time_limit < state[agent_id]['time']:
            return True
        return False

    def reset(self):
        self._last_section = -1
        self._current_lap = 0


tasks = {
    'time_based': TimeBasedRacingTask,
    'rank_discounted': RankDiscountedProgressTask,
    'segment_progress': ProgressTaskWtPenalty
}


def task_from_spec(spec: TaskSpec) -> Task:
    if spec.task_name in tasks:
        return tasks[spec.task_name](**spec.params)


def register_task(name: str, task: Task):
    tasks[name] = task