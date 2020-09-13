from dataclasses import dataclass

import gym
import numpy as np

from racecar_gym.models import simulation


class Sensor:

    def space(self) -> gym.Space:
        pass

    def observe(self) -> np.ndarray:
        pass


class FixedTimestepSensor(Sensor):

    def __init__(self, sensor: Sensor, frequency: int, time_step: float):
        self._sensor = sensor
        self._frequency = 1 / frequency
        self._time_step = time_step
        self._last_timestep = self._frequency
        self._last_observation = None

    def space(self) -> gym.Space:
        return self._sensor.space()

    def observe(self) -> np.ndarray:
        self._last_timestep += self._time_step
        if self._last_timestep >= self._frequency:
            self._last_observation = self._sensor.observe()
            self._last_timestep = 0
        return self._last_observation


class SimulatedSensor(Sensor):
    def __init__(self, handle: simulation.SimulationHandle):
        self._handle = handle

class Lidar(SimulatedSensor):

    @dataclass
    class Config:
        rays: int
        range: float
        min_range: float

    def __init__(self, handle: simulation.SimulationHandle, config: Config):
        super().__init__(handle)
        self._config = config
        self._raycast = simulation.RayCast(rays=config.rays, min_distance=config.min_range, scan_range=config.range)

    def space(self) -> gym.Space:
        return gym.spaces.Box(low=self._config.min_range,
                              high=self._config.min_range + self._config.range,
                              dtype=np.float32,
                              shape=(self._config.rays,))

    def observe(self) -> np.ndarray:
        return self._raycast.scan_from_link(body_id=self._handle.body_id, index=self._handle.link_index)


class RGBCamera(SimulatedSensor):

    @dataclass
    class Config:
        width: int
        height: int
        fov: int
        distance: float

    def __init__(self, handle: simulation.SimulationHandle, config: Config):
        super().__init__(handle)
        self._config = config
        self._camera = simulation.Camera(target_distance=config.distance, fov=config.fov)

    def space(self) -> gym.Space:
        return gym.spaces.Box(low=0,
                              high=255,
                              shape=(self._config.height, self._config.width, 3),
                              dtype=np.uint8)

    def observe(self) -> np.ndarray:
        pose = simulation.get_pose(self._handle.body_id, self._handle.link_index)
        self._camera.set_pose(pose=pose)
        return self._camera.get_image(height=self._config.height, width=self._config.width)


class InertialMeasurementUnit(SimulatedSensor):

    @dataclass
    class Config:
        max_acceleration: float
        max_angular_velocity: float

    def __init__(self, handle: simulation.SimulationHandle, config: Config):
        super().__init__(handle)
        self._config = config
        self._last_velocity = simulation.get_velocity(id=handle.body_id)

    def space(self) -> gym.Space:
        high = np.array(3 * [self._config.max_acceleration] + 3 * [self._config.max_angular_velocity])
        low = -high
        return gym.spaces.Box(low=low, high=high)

    def observe(self) -> np.ndarray:
        velocity = np.array(simulation.get_velocity(id=self._handle.body_id))
        linear_acceleration = (velocity[:3] - self._last_velocity[:3]) / 0.01
        self._last_velocity = velocity
        return np.append(linear_acceleration, velocity[3:])


class Tachometer(SimulatedSensor):

    @dataclass
    class Config:
        max_linear_velocity: float
        max_angular_velocity: float

    def __init__(self, handle: simulation.SimulationHandle, config: Config):
        super().__init__(handle)
        self._config = config

    def space(self) -> gym.Space:
        high = np.array(3 * [self._config.max_linear_velocity] + 3 * [self._config.max_angular_velocity])
        low = -high
        return gym.spaces.Box(low=low, high=high)

    def observe(self) -> np.ndarray:
        return np.array(simulation.get_velocity(id=self._handle.body_id))

class GPS(SimulatedSensor):

    @dataclass
    class Config:
        max_x: float
        max_y: float
        max_z: float

    def __init__(self, handle: simulation.SimulationHandle, config: Config):
        super().__init__(handle)
        self._config = config

    def space(self) -> gym.Space:
        high = np.array([self._config.max_x, self._config.max_y, self._config.max_z] + 3 * [np.pi])
        low = -high
        return gym.spaces.Box(low=low, high=high)

    def observe(self) -> np.ndarray:
        position, orientation = simulation.get_pose(id=self._handle.body_id)
        return np.append(position, orientation)
