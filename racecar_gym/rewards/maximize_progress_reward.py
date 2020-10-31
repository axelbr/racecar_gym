from racecar_gym.core.tasks import Task, RewardRange

class MaximizeProgressTask(Task):
    def __init__(self, laps: int, time_limit: float, terminate_on_collision: bool,
                 delta_progress=0.01, collision_reward=-100, frame_reward=-0.1, progress_reward=10):
        self._time_limit = time_limit
        self._laps = laps
        self._terminate_on_collision = terminate_on_collision
        self._last_stored_progress = None
        # reward params
        self._delta_progress = delta_progress
        self._progress_reward = progress_reward
        self._collision_reward = collision_reward
        self._frame_reward = frame_reward

    def reward_range(self) -> RewardRange:
        pass

    def reward(self, agent_id, state, action) -> float:
        progress = state[agent_id]['lap'] + state[agent_id]['progress']
        if self._last_stored_progress is None:
            self._last_stored_progress = progress
        delta = progress - self._last_stored_progress

        reward = self._frame_reward
        if state[agent_id]['collision'] == True:
            reward += self._collision_reward
        elif delta > self._delta_progress:
            reward += self._progress_reward
            self._last_stored_progress = progress
        return reward

    def done(self, agent_id, state) -> bool:
        if self._terminate_on_collision and state[agent_id]['collision']:
            return True
        return state[agent_id]['lap'] > self._laps or self._time_limit < state[agent_id]['time']

    def reset(self):
        self._last_stored_progress = None