from .task import Task
import numpy as np


class WaypointFollow(Task):
    def __init__(self, laps: int, time_limit: float, terminate_on_collision: bool, n_min_rays: int = 1080,
                 collision_reward: float = 0.0, state_gain: float = 0.1, action_gain: float = 0.5):
        self._time_limit = time_limit
        self._laps = laps
        self._terminate_on_collision = terminate_on_collision
        self._n_min_rays_termination = n_min_rays  # collision also when exist `n_min_rays` <= safe threshold (e.g.0.25m)
        self._last_action = {'motor': 0.0, 'steering': 0.0}
        # reward params
        self._collision_reward = collision_reward
        self._state_gain = state_gain
        self._action_gain = action_gain

    def reward(self, agent_id, state, action) -> float:
        """
        Idea: def. a quadratic cost by weighting the deviation from a target state (waypoint) and from the prev action.
        However, aiming to have a positive reward, the change the sign (i.e. reward=-cost) lead to cumulative penalties
        which encourage the agent to terminate the episode asap.
        For this reason, the resulting negative cost is passed through an exponential function,
        obtaining the desired behaviour:
            1. exp(- small cost) -> 1
            2. exp(- big cost) -> 0
        Optionally, we can add a negative reward in case of collision.
        """
        agent_state = state[agent_id]
        position = agent_state['pose'][:3]
        waypoint = agent_state['next_waypoint']
        Q = self._state_gain * np.identity(len(position))
        R = self._action_gain * np.identity(len(action))
        delta_pos = waypoint - position
        delta_act = np.array(list(action.values())) - np.array(list(self._last_action.values()))
        cost = (np.matmul(delta_pos, np.matmul(Q, delta_pos)) + np.matmul(delta_act, np.matmul(R, delta_act)))
        reward = np.exp(-cost)
        if self._check_collision(agent_state):
            reward += self._collision_reward
        self._last_action = action
        return reward

    def done(self, agent_id, state) -> bool:
        agent_state = state[agent_id]
        if self._terminate_on_collision and self._check_collision(agent_state):
            return True
        return agent_state['lap'] > self._laps or self._time_limit < agent_state['time']

    def _check_collision(self, agent_state):
        safe_margin = 0.25
        collision = agent_state['wall_collision'] or len(agent_state['opponent_collisions']) > 0
        if 'observations' in agent_state and 'lidar' in agent_state['observations']:
            n_min_rays = sum(np.where(agent_state['observations']['lidar'] <= safe_margin, 1, 0))
            return n_min_rays > self._n_min_rays_termination or collision
        return collision

    def reset(self):
        self._last_stored_progress = None
