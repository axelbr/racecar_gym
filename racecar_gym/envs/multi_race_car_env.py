import math
from typing import List, Dict, Tuple, Any, Callable, Generic

import gym
import numpy as np
from pybullet_utils.bullet_client import BulletClient

from racecar_gym.definitions import Pose
from racecar_gym.envs.definitions import Action, StepReturn, State, Observation
from racecar_gym.envs.specs import ScenarioSpec
from racecar_gym.envs.tasks import from_spec, Task
from racecar_gym.models import load_map_from_config, load_map_by_name, load_vehicle_from_config, Map, \
    load_vehicle_by_name

from racecar_gym.models.racecar import RaceCar


class Agent(Generic[Observation, Action]):

    def __init__(self, vehicle: RaceCar, task: Task, sensors: List[str], time_step: float):
        self._vehicle = vehicle
        self._task = task
        self._available_sensors = sensors
        self._time = 0.0
        self._time_step = time_step

    @property
    def action_space(self) -> gym.Space:
        return self._vehicle.action_space

    @property
    def observation_space(self) -> gym.Space:
        return self._vehicle.space()

    def step(self, action: Action) -> StepReturn:
        self._time += self._time_step
        observation, info = self._vehicle.observe(sensors=self._available_sensors)
        self._vehicle.step(velocity=action[0], steering_angle=action[1], force=action[2])
        observation['time'] = self._time
        done = self._task.done(observation)
        reward = self._task.reward(observation, action)
        return observation, reward, done, info

    def reset(self, pose: Pose) -> Observation:
        self._vehicle.reset(pose)
        self._time = 0
        observation, info = self._vehicle.observe(sensors=self._available_sensors)
        return observation


class MultiRaceCarEnv(gym.Env):

    def __init__(self, client_factory: Callable[[], BulletClient], scenario: ScenarioSpec):
        self._scenario = scenario
        self._client = client_factory()
        self._map = self._load_map()
        self._simulation_time = 0.0
        self._agents = self._init_agents(map=self._map, scenario=scenario)

    def _init_agents(self, map: Map, scenario: ScenarioSpec) -> List[Agent[Dict, np.ndarray]]:
        agents = []
        for agent_spec in self._scenario.agents:
            if agent_spec.vehicle.config_file:
                vehicle = load_vehicle_from_config(client=self._client, map=map, config_file=agent_spec.vehicle.config_file)
            elif agent_spec.vehicle.name:
                vehicle = load_vehicle_by_name(client=self._client, map=map, vehicle=agent_spec.vehicle.name)
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
            obs = agent.reset(pose=self._map.starting_pose(position=i))
            initial_obs.append(obs)

        self.action_space = self._make_action_space(self._agents)
        self.observation_space = self._make_observation_space(self._scenario)

        return initial_obs

    def render(self, mode='human'):
        pass

    def _observe(self, vehicle: RaceCar, sensors: List[str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        observation, info = vehicle.observe(sensors=sensors)
        observation['time'] = self._simulation_time
        return observation, info

    def _make_action_space(self, agents: List[Agent]) -> gym.spaces.Tuple:
        return gym.spaces.Tuple([a.action_space for a in agents])

    def _make_observation_space(self, scenario: ScenarioSpec) -> gym.Space:
        spaces = []
        for agent in self._agents:
            space = {}

            observation_space = agent.observation_space
            space.update(observation_space.spaces.items())

            lower_bound, upper_bound = self._map.bounds
            space['pose'] = gym.spaces.Box(
                low=np.array([*lower_bound, -math.pi, -math.pi, -math.pi]),
                high=np.array([*upper_bound, math.pi, math.pi, math.pi])
            )

            # 30 m/s and 5*3.14 rad/s seem to be reasonable bounds
            space['velocity'] = gym.spaces.Box(
                low=np.array([-30.0, -30.0, -30.0, -10 * math.pi, -10 * math.pi, -10 * math.pi]),
                high=np.array([30.0, 30.0, 30.0, 10 * math.pi, 10 * math.pi, 10 * math.pi])
            )

            space['time'] = gym.spaces.Box(low=0, high=scenario.max_time, shape=(1,))
            space['collision'] = gym.spaces.Discrete(2)
            space['lap'] = gym.spaces.Discrete(scenario.laps)

            spaces.append(gym.spaces.Dict(space))

        return gym.spaces.Tuple(spaces)
