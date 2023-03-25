from typing import Dict, SupportsFloat, Any, Tuple, Optional
import gymnasium
import numpy as np
from gymnasium.core import ActType, ObsType

from racecar_gym.envs.scenarios import MultiAgentScenario
from racecar_gym.core.definitions import Pose


class MultiAgentRaceEnv(gymnasium.Env):

    metadata = {
        'render_modes': ['human', 'rgb_array_follow', 'rgb_array_birds_eye', 'rgb_array_lidar']
    }

    def __init__(self, scenario: str, render_mode: str = 'human', render_options: Dict = None):
        self._scenario = MultiAgentScenario.from_spec(scenario, rendering=render_mode == 'human')
        self._initialized = False
        assert render_mode in self.metadata['render_modes'], f'Invalid render mode: {render_mode}'
        self._render_mode = render_mode
        self._render_options = render_options or {}
        if not 'agent' in self._render_options:
            self._render_options['agent'] = next(iter(self._scenario.agents))
        self._time = 0.0
        self.action_space = gymnasium.spaces.Dict([(k, a.action_space) for k, a in self._scenario.agents.items()])

    @property
    def scenario(self):
        return self._scenario

    @property
    def observation_space(self):
        spaces = {}
        for id, agent in self._scenario.agents.items():
            spaces[id] = agent.observation_space
            spaces[id].spaces['time'] = gymnasium.spaces.Box(low=0, high=1, shape=())
        return gymnasium.spaces.Dict(spaces)

    def step(self, action: ActType) -> Tuple[ObsType, Dict[str, SupportsFloat], Dict[str, bool], bool, Dict[str, Any]]:

        assert self._initialized, 'Reset before calling step'

        observations = {}
        dones = {}
        rewards = {}
        state = {}

        for id, agent in self._scenario.agents.items():
            observations[id], state[id] = agent.step(action=action[id])

        self._scenario.world.update()
        state = self._scenario.world.state()

        for id, agent in self._scenario.agents.items():
            state[id]['observations'] = observations[id]
            observations[id]['time'] = np.array(state[id]['time'], dtype=np.float32)
            dones[id] = agent.done(state)
            rewards[id] = agent.reward(state, action[id])

        return observations, rewards, dones, False, state

    def set_state(self, agent: str, pose: Pose):
        self._scenario.agents[agent].reset(pose=pose)

    def reset(self, *, seed: Optional[int] = None, options: Dict[str, Any] = None) -> Tuple[ObsType, Dict[str, Any]]:
        super().reset(seed=seed, options=options)
        if not self._initialized:
            self._scenario.world.init()
            self._initialized = True
        if options is not None:
            mode = options.get('mode', 'grid')
        else:
            mode = 'grid'

        observations = {}
        for agent in self._scenario.agents.values():
            obs = agent.reset(self._scenario.world.get_starting_position(agent=agent, mode=mode))
            observations[agent.id] = obs
        self._scenario.world.reset()
        self._scenario.world.update()
        state = self._scenario.world.state()
        for agent in self._scenario.agents.values():
            observations[agent.id]['time'] = np.array(state[agent.id]['time'], dtype=np.float32)
        return observations, state

    def render(self):
        if self._render_mode == 'human':
            return None
        else:
            options = self._render_options.copy()
            mode = self._render_mode.replace('rgb_array_', '')
            agent = options.pop('agent')
            return self._scenario.world.render(agent_id=agent, mode=mode, **options)

