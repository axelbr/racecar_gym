from typing import List, Dict, Callable

import gym
import numpy as np
from pybullet_utils.bullet_client import BulletClient

from racecar_gym.bullet import load_vehicle_from_config, load_vehicle_by_name, load_map_from_config, load_map_by_name
from racecar_gym.models.agent import Agent
from racecar_gym.models.map import Map
from racecar_gym.models.specs import ScenarioSpec
from racecar_gym.models.tasks import from_spec


class MultiRaceCarEnv(gym.Env):

    def __init__(self, client_factory: Callable[[], BulletClient], scenario: ScenarioSpec):
        self._scenario = scenario
        self._client = client_factory()
        self._map = self._load_map()
        self._simulation_time = 0.0
        self._agents = self._init_agents(map=self._map, scenario=scenario)


    def _init_agents(self, map: Map, scenario: ScenarioSpec) -> List[Agent[Dict, np.ndarray]]:
        agents = []
        for i, agent_spec in enumerate(self._scenario.agents):
            if agent_spec.vehicle.config_file:
                vehicle = load_vehicle_from_config(client=self._client, map=map, config_file=agent_spec.vehicle.config_file)
            elif agent_spec.vehicle.name:
                vehicle = load_vehicle_by_name(client=self._client, map=map, vehicle=agent_spec.vehicle.name, id=i)
            else:
                raise ValueError('You need to specify either a config file or a valid vehicle name.')

            agent = Agent[Dict, np.ndarray](vehicle=vehicle, task=from_spec(spec=agent_spec.task), sensors=agent_spec.vehicle.sensors, time_step=scenario.simulation.time_step)
            agents.append(agent)
        return agents


    def _load_map(self) -> Map:
        if self._scenario.map.config_file:
            map = load_map_from_config(client=self._client, config_file=self._scenario.map.config_file)
        elif self._scenario.map.name:
            map = load_map_by_name(client=self._client, map_name=self._scenario.map.name)
        else:
            raise ValueError('You need to specify either a config file or a valid map name.')

        return map

    def step(self, action: np.ndarray):
        assert len(action) == len(self._agents), f'An action must be provided for every vehicle.'

        observations = []
        dones = []
        rewards = []
        infos = []

        for i, agent in enumerate(self._agents):
            observation, reward, done, info = agent.step(action=action[i])
            observations.append(observation)
            dones.append(done)
            rewards.append(reward)
            infos.append(info)

        self._client.stepSimulation()
        self._simulation_time += self._scenario.simulation.time_step
        return observations, rewards, dones, infos

    def reset(self):
        self._client.resetSimulation()
        self._client.setGravity(0, 0, -9.81)
        self._client.setTimeStep(self._scenario.simulation.time_step)
        self._simulation_time = 0.0

        self._map.reset()
        initial_obs = []
        for i, agent in enumerate(self._agents):
            obs = agent.reset()
            initial_obs.append(obs)

        self.action_space = self._make_action_space(self._agents)
        self.observation_space = self._make_observation_space(self._scenario)

        return initial_obs

    def render(self, mode='human'):
        pass

    def _make_action_space(self, agents: List[Agent]) -> gym.spaces.Tuple:
        return gym.spaces.Tuple([a.action_space for a in agents])

    def _make_observation_space(self, scenario: ScenarioSpec) -> gym.Space:
        spaces = []
        for agent in self._agents:
            space = {}

            observation_space = agent.observation_space
            space.update(observation_space.spaces.items())
            space['time'] = gym.spaces.Box(low=0, high=scenario.max_time, shape=(1,))
            space['collision'] = gym.spaces.Discrete(2)
            space['lap'] = gym.spaces.Discrete(scenario.laps)

            spaces.append(gym.spaces.Dict(space))

        return gym.spaces.Tuple(spaces)
