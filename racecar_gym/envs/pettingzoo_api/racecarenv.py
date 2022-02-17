import functools
from typing import Dict

import gym
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector

from ..scenarios import MultiAgentScenario


class _MultiAgentEnv(AECEnv):

    metadata = {'render.modes': ['follow', 'birds_eye'], "name": "racecar_v1"}

    def __init__(self, scenario: MultiAgentScenario, reset_mode='grid'):
        self._scenario = scenario
        self._reset_mode = reset_mode
        self.possible_agents = sorted([a for a in scenario.agents])
        self.agents = self.possible_agents[:]
        self._agent_selector = agent_selector(self.possible_agents)
        self.agent_selection = self.possible_agents[0]

        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: {} for agent in self.agents}
        self.observations = {agent: {} for agent in self.agents}
        self.actions = {agent: {} for agent in self.agents}
        self.num_moves = 0

        self._initialized = False
        self._scenario.world.init()
        super().__init__()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> gym.Space:
        return self._scenario.agents[agent].observation_space

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> gym.Space:
        return self._scenario.agents[agent].action_space

    def observe(self, agent):
        return dict(self.observations[agent])

    def reset(self):
        self.agents = self.possible_agents.copy()
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        for agent_id in self.agents:
            agent = self._scenario.agents[agent_id]
            start_position = self._scenario.world.get_starting_position(agent=agent, mode=self._reset_mode)
            obs = agent.reset(pose=start_position)
            self.observations[agent_id] = obs
            self.rewards[agent_id] = 0
            self._cumulative_rewards[agent_id] = 0
            self.dones[agent_id] = False
            self.infos[agent_id] = {}
            self.observations[agent_id] = self._scenario.agents[agent_id].vehicle.observe()
            self.actions[agent_id] = {}
            self.num_moves = 0

        self._scenario.world.reset()
        self._scenario.world.update()
        self.state = self._scenario.world.state()


    def step(self, action: Dict):
        if self.dones[self.agent_selection]:
            self._was_done_step(action)
            if len(self.agents) > 0:
                self.agent_selection = self._agent_selector.next()
            return

        car = self._scenario.agents[self.agent_selection].vehicle
        car.control(action)
        self._cumulative_rewards[self.agent_selection] = 0
        self.actions[self.agent_selection] = action

        if self._agent_selector.is_last():
            self._scenario.world.update()
            self.state = self._scenario.world.state()
            self.num_moves += 1
            for agent_id in self.agents:
                agent = self._scenario.agents[agent_id]
                task = agent.task
                self.rewards[agent_id] = task.reward(agent_id=agent_id, state=self.state, action=self.actions[agent_id])
                self.dones[agent_id] = task.done(agent_id=agent_id, state=self.state)
                self.observations[agent_id] = agent.vehicle.observe()
        else:
            self._clear_rewards()
        if len(self.agents) > 0:
            self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()


    def render(self, mode='follow', **kwargs):
        if mode not in _MultiAgentEnv.metadata['render.modes']:
            raise NotImplementedError(
                f'"{mode}" is no supported render mode. Available render modes: {_MultiAgentEnv.metadata["render.modes"]}')
        return self._scenario.world.render(agent_id=self.agent_selection, mode=mode, **kwargs)

    def close(self):
        pass



