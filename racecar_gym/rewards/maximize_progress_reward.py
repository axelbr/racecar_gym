from racecar_gym.core.tasks import Task, RewardRange

class MaximizeProgressTask(Task):
    def __init__(self, laps: int, time_limit: float, terminate_on_collision: bool):
        self._time_limit = time_limit
        self._laps = laps
        self._terminate_on_collision = terminate_on_collision
        self._previous_progress = None

    def reward_range(self) -> RewardRange:
        pass

    def reward(self, agent_id, state, action) -> float:
        progress = state[agent_id]['lap'] + state[agent_id]['progress']
        if self._previous_progress is None:
            delta = 0.0
        else:
            delta = progress - self._previous_progress
        self._previous_progress = progress
        reward = 0
        if state[agent_id]['collision'] == True:
            reward -= 1
        elif delta > 0:
            reward += 1
        else:
            reward -= 0.01
        return reward

    def done(self, agent_id, state) -> bool:
        if self._terminate_on_collision and state[agent_id]['collision']:
            return True
        return state[agent_id]['lap'] > self._laps or self._time_limit < state[agent_id]['time']

    def reset(self):
        self._previous_progress = None