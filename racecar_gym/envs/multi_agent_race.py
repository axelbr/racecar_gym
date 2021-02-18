import math
from typing import Dict, Union
import gym
from .scenarios import MultiAgentScenario
from ..core.definitions import Pose, Velocity


class MultiAgentRaceEnv(gym.Env):

    metadata = {'render.modes': ['follow', 'birds_eye', 'lidar']}

    def __init__(self, scenario: MultiAgentScenario):
        self._scenario = scenario
        self._initialized = False
        self._time = 0.0
        self.observation_space = gym.spaces.Dict([(k, a.observation_space) for k, a in scenario.agents.items()])
        self.action_space = gym.spaces.Dict([(k, a.action_space) for k, a in scenario.agents.items()])

    @property
    def scenario(self):
        return self._scenario

    def step(self, action: Dict):

        assert self._initialized, 'Reset before calling step'

        observations = {}
        dones = {}
        rewards = {}

        state = self._scenario.world.state()
        for id, agent in self._scenario.agents.items():
            observation, info = agent.step(action=action[id])
            state[id]['observations'] = observation
            done = agent.done(state)
            reward = agent.reward(state, action[id])
            observations[id] = observation
            dones[id] = done
            rewards[id] = reward
        self._time = self._scenario.world.update()
        return observations, rewards, dones, state

    def set_state(self, agent: str, pose: Pose):
        self._scenario.agents[agent].reset(pose=pose)

    def reset(self, mode: str = 'grid'):
        if not self._initialized:
            self._scenario.world.init()
            self._initialized = True

        observations = {}
        for agent in self._scenario.agents.values():
            obs = agent.reset(self._scenario.world.get_starting_position(agent=agent, mode=mode))
            observations[agent.id] = obs
            self._scenario.world.reset()
            self._scenario.world.update()

        return observations

    def render(self, mode='follow', agent: str = None, **kwargs):
        if mode not in MultiAgentRaceEnv.metadata['render.modes']:
            raise NotImplementedError(f'"{mode}" is no supported render mode. Available render modes: {MultiAgentRaceEnv.metadata["render.modes"]}')
        if agent is None:
            agent = list(self._scenario.agents.keys())[0]
        return self._scenario.world.render(agent_id=agent, mode=mode, **kwargs)

