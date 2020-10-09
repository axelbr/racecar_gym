from typing import Dict

import gym

from racecar_gym.envs.scenarios import MultiAgentScenario


class MultiAgentRaceCarEnv(gym.Env):

    def __init__(self, scenario: MultiAgentScenario):
        self._scenario = scenario
        self._initialized = False
        self._time = 0.0
        self.observation_space = gym.spaces.Dict([(k, a.observation_space) for k, a in scenario.agents.items()])
        self.action_space = gym.spaces.Dict([(k, a.action_space) for k, a in scenario.agents.items()])

    def step(self, action: Dict):

        assert self._initialized, 'Reset before calling step'

        observations = {}
        dones = {}
        rewards = {}
        state = self._scenario.world.state()
        for id, agent in self._scenario.agents.items():
            observation, info = agent.step(action=action[id])
            done = agent.done(state)
            reward = agent.reward(state, action[id])

            observations[id] = observation
            dones[id] = done
            rewards[id] = reward

        self._time = self._scenario.world.update()

        return observations, rewards, dones, state

    def reset(self):
        if not self._initialized:
            self._scenario.world.init()
            self._initialized = True

        observations = {}
        for agent in self._scenario.agents.values():
            obs = agent.reset(self._scenario.world.get_starting_position(agent))
            obs['time'] = 0
            observations[agent.id] = obs
        self._scenario.world.reset()
        return observations
