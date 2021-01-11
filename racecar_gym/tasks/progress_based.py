from .task import Task
import numpy as np


class MaximizeProgressTask(Task):
  def __init__(self, laps: int, time_limit: float, terminate_on_collision: bool,
               delta_progress: float = 0.0, collision_reward: float = 0.0,
               frame_reward: float = 0.0, progress_reward: float = 100.0):
    self._time_limit = time_limit
    self._laps = laps
    self._terminate_on_collision = terminate_on_collision
    self._last_stored_progress = None
    # reward params
    self._delta_progress = delta_progress
    self._progress_reward = progress_reward
    self._collision_reward = collision_reward
    self._frame_reward = frame_reward

  def reward(self, agent_id, state, action) -> float:
    agent_state = state[agent_id]
    progress = agent_state['lap'] + agent_state['progress']
    if self._last_stored_progress is None:
      self._last_stored_progress = progress
    delta = abs(progress - self._last_stored_progress)
    if delta > .5:  # the agent is crossing the starting line in the wrong direction
      delta = (1-progress) + self._last_stored_progress
    reward = self._frame_reward
    collision = agent_state['wall_collision'] or len(agent_state['opponent_collisions']) > 0
    if collision == True:
      reward += self._collision_reward
    reward += delta * self._progress_reward
    self._last_stored_progress = progress
    return reward

  def done(self, agent_id, state) -> bool:
    agent_state = state[agent_id]
    collision = agent_state['wall_collision'] or len(agent_state['opponent_collisions']) > 0
    if self._terminate_on_collision and collision:
      return True
    return agent_state['lap'] > self._laps or self._time_limit < agent_state['time']

  def reset(self):
    self._last_stored_progress = None


class MaximizeProgressMaskObstacleTask(MaximizeProgressTask):
  def __init__(self, laps: int, time_limit: float, terminate_on_collision: bool, delta_progress=0.0,
               collision_reward=0, frame_reward=0, progress_reward=100):
    super().__init__(laps, time_limit, terminate_on_collision, delta_progress, collision_reward, frame_reward,
                     progress_reward)

  def reward(self, agent_id, state, action) -> float:
    progress_reward = super().reward(agent_id, state, action)
    distance_to_obstacle = state[agent_id]['obstacle']
    if distance_to_obstacle < .5:   # max distance = 1, meaning perfectly centered in the widest point of the track
      return 0.0
    else:
      return progress_reward


class RankDiscountedMaximizeProgressTask(MaximizeProgressTask):
  def __init__(self, laps: int, time_limit: float, terminate_on_collision: bool, delta_progress=0.001,
               collision_reward=-100, frame_reward=-0.1, progress_reward=1):
    super().__init__(laps, time_limit, terminate_on_collision, delta_progress, collision_reward, frame_reward,
                     progress_reward)

  def reward(self, agent_id, state, action) -> float:
    rank = state[agent_id]['rank']
    reward = super().reward(agent_id, state, action)
    reward = reward / float(rank)
    return reward
