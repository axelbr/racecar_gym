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
        infos = {}

        for id, agent in self._scenario.agents.items():
            observation, info = agent.step(action=action[id])
            observation.update(self._scenario.world.state())

            done = agent.done(observation)
            reward = agent.reward(observation, action[id])

            observations[id] = observation
            dones[id] = done
            rewards[id] = reward
            infos[id] = info

        self._time = self._scenario.world.update()

        return observations, rewards, dones, infos

    def reset(self):
        if not self._initialized:
            self._scenario.world.init()
            self._initialized = True
        else:
            self._scenario.world.reset()
        observations = {}
        for i, agent in enumerate(self._scenario.agents.values()):
            obs = agent.reset(self._scenario.world.initial_pose(i))
            obs['time'] = 0
            observations[agent.id] = obs
        return observations
