import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Union, Callable

import gym
import numpy as np
import pybullet
from pybullet_utils.bullet_client import BulletClient

from racecar_gym.models import load_map, load_vehicle
from racecar_gym.models.racecar import RaceCar


@dataclass
class MultiRaceCarScenario:
    map: str
    cars: Union[List[str], str]
    time_step: float = 0.01
    no_of_cars: int = 1
    max_speed: float = 14.0
    max_steering_angle: float = 0.4
    lidar_rays: int = 100
    lidar_range: float = 5
    start_gui: bool = False
    laps: int = 2
    max_time: float = 120.0


class MultiRaceCarEnv(gym.Env):

    def __init__(self, client_factory: Callable[[], BulletClient], scenario: MultiRaceCarScenario):
        self._scenario = scenario
        self._client = client_factory()

    def _make_action_space(self, scenario: MultiRaceCarScenario) -> gym.Space:
        space = np.array([-scenario.max_speed, -scenario.max_steering_angle])
        action_spaces = [gym.spaces.Box(low=-space, high=space) for _ in range(scenario.no_of_cars)]
        return gym.spaces.Tuple(action_spaces)

    def _make_observation_space(self, scenario: MultiRaceCarScenario) -> gym.Space:
        spaces = []
        for _ in self._vehicles:
            space = {}
            if scenario.lidar_rays > 0:
                space['lidar'] = gym.spaces.Box(
                    low=0,
                    high=scenario.lidar_range,
                    shape=(scenario.lidar_rays,)
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

    def step(self, action: np.ndarray):
        assert len(action) == len(self._vehicles), f'An action must be provided for every vehicle.'

        observations = []
        dones = []
        rewards = []

        for i, vehicle in enumerate(self._vehicles):
            observation, reward, done = self._step(vehicle, action[i])
            observations.append(observation)
            dones.append(done)
            rewards.append(reward)

        self._client.stepSimulation()
        self._simulation_time += self._scenario.time_step
        return tuple(observations), tuple(rewards), tuple(dones), tuple([{} for _ in self._vehicles])

    def _step(self, vehicle: RaceCar, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool]:
        observation = self._observe(vehicle)
        vehicle.step(velocity=action[0], steering_angle=action[1], force=action[2])
        self._update_lap(vehicle)
        done = self._check_termination(vehicle, observation)
        reward = 0.0
        return observation, reward, done

    def _observe(self, vehicle: RaceCar) -> Dict[str, Any]:
        observation = vehicle.observe(sensors=['odometry', 'lidar'])
        observation['collision'] = self._check_collision(vehicle)
        observation['time'] = self._simulation_time
        return observation

    def _check_collision(self, vehicle: RaceCar) -> bool:
        collisions = set([c[2] for c in self._client.getContactPoints(vehicle.id)])

        # check if vehicle crashed into the wall
        if self._map.walls_id in collisions:
            print(f'Car {vehicle.id} crashed into wall')
            return True

        # check if vehicle crashed into an opponent
        if not set([v.id for v in self._vehicles]).isdisjoint(collisions):
            print(f'Car {vehicle.id} crashed into car {collisions}')
            return True

        return False

    def _check_termination(self, vehicle: RaceCar, observation: Dict[str, Any]) -> bool:
        if observation['collision']:
            return True
        if vehicle.lap > self._scenario.laps:
            return True
        if self._simulation_time >= self._scenario.max_time:
            return True
        return False

    def _update_lap(self, vehicle: RaceCar):
        closest_points = self._client.getClosestPoints(vehicle.id, self._map.finish_id, 0.05)
        if len(closest_points) > 0:
            if not vehicle.on_finish:
                vehicle.on_finish = True
                vehicle.lap += 1
        else:
            if vehicle.on_finish:
                vehicle.on_finish = False

    def reset(self):
        self._client.resetSimulation()
        self._client.setGravity(0, 0, -9.81)
        self._client.setTimeStep(self._scenario.time_step)
        map = load_map(client=self._client, config_file=self._scenario.map)

        if type(self._scenario.cars) is List:
            vehicles = [
                load_vehicle(pose=map.starting_pose(i), client=self._client, config_file=car_config)
                for i, car_config
                in enumerate(self._scenario.cars)
            ]
        else:
            vehicles = [
                load_vehicle(pose=map.starting_pose(i), client=self._client, config_file=self._scenario.cars)
                for i
                in range(self._scenario.no_of_cars)
            ]

        self._map = map
        self._vehicles = vehicles
        self._simulation_time = 0.0
        self.action_space = self._make_action_space(self._scenario)
        self.observation_space = self._make_observation_space(self._scenario)
        return [self._observe(v) for v in self._vehicles]

    def render(self, mode='human'):
        pass
