import math
from typing import List, Dict, Tuple, Any, Callable

import gym
import numpy as np
from pybullet_utils.bullet_client import BulletClient

from racecar_gym.envs.specs import ScenarioSpec
from racecar_gym.envs.tasks import from_spec
from racecar_gym.models import load_map, load_vehicle, Map
from racecar_gym.models.racecar import RaceCar


class MultiRaceCarEnv(gym.Env):

    def __init__(self, client_factory: Callable[[], BulletClient], scenario: ScenarioSpec):
        self._scenario = scenario
        self._client = client_factory()
        self._map, self._vehicles = self._load_models()
        self._simulation_time = 0.0
        self._task = from_spec(spec=scenario.task_spec)

    def _load_models(self) -> Tuple[Map, List[RaceCar]]:
        map = load_map(client=self._client, config_file=self._scenario.map_spec.config_file)
        vehicles = [
            load_vehicle(client=self._client, map=map, config_file=vehicle_spec.config_file)
            for i, vehicle_spec
            in enumerate(self._scenario.vehicle_spec)
        ]
        return map, vehicles


    def step(self, action: np.ndarray):
        assert len(action) == len(self._vehicles), f'An action must be provided for every vehicle.'

        observations = []
        dones = []
        rewards = []
        infos = []

        for i in range(len(self._vehicles)):
            observation, reward, done, info = self._step_vehicle(vehicle_index=i, action=action[i])
            observations.append(observation)
            dones.append(done)
            rewards.append(reward)
            infos.append(info)

        self._client.stepSimulation()
        self._simulation_time += self._scenario.simulation_spec.time_step
        return observations, rewards, dones, infos

    def _step_vehicle(self, vehicle_index: int, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        vehicle = self._vehicles[vehicle_index]
        spec = self._scenario.vehicle_spec[vehicle_index]
        observation, info = self._observe(vehicle=vehicle, sensors=spec.sensors)
        vehicle.step(velocity=action[0], steering_angle=action[1], force=action[2])
        done = self._task.done(observation)
        reward = self._task.reward(observation, action)
        return observation, reward, done, info

    def reset(self):
        self._client.resetSimulation()
        self._client.setGravity(0, 0, -9.81)
        self._client.setTimeStep(self._scenario.simulation_spec.time_step)
        self._simulation_time = 0.0

        self._map.reset()
        for i, vehicle in enumerate(self._vehicles):
            vehicle.reset(pose=self._map.starting_pose(position=i))

        self.action_space = self._make_action_space(self._vehicles)
        self.observation_space = self._make_observation_space(self._scenario)

        return [self._observe(v, sensors=self._scenario.vehicle_spec[i].sensors)[0] for i, v in enumerate(self._vehicles)]

    def render(self, mode='human'):
        pass

    def _observe(self, vehicle: RaceCar, sensors: List[str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        observation, info = vehicle.observe(sensors=sensors)

        if self._map.walls_id in info['collisions']:
            print(f'Car {vehicle.id} crashed into wall')

        if not (set([v.id for v in self._vehicles]) - {self._map.walls_id}).isdisjoint(info['collisions']):
            print(f'Car {vehicle.id} crashed into car {observation["collisions"]}')

        observation['time'] = self._simulation_time
        return observation, info

    def _make_action_space(self, vehicles: List[RaceCar]) -> gym.spaces.Tuple:
        return gym.spaces.Tuple([v.action_space for v in vehicles])

    def _make_observation_space(self, scenario: ScenarioSpec) -> gym.Space:
        spaces = []
        for vehicle in self._vehicles:
            space = {}

            if vehicle.config.sensors.lidar:
                lidar_config = vehicle.config.sensors.lidar
                space['lidar'] = gym.spaces.Box(
                    low=0,
                    high=lidar_config.range,
                    shape=(lidar_config.rays,)
                )

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
