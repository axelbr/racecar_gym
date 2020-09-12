import math
from dataclasses import dataclass

import gym
import numpy as np

from racecar_gym.models.simulation import SimulationHandle

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
    def __init__(self, handle: SimulationHandle):
        self._handle = handle

class Lidar(SimulatedSensor):

    @dataclass
    class Config:
        rays: int
        range: float
        min_range: float

    def __init__(self, handle: SimulationHandle, config: Config):
        super().__init__(handle)
        self._config = config
        self._hit_color = [1, 0, 0]
        self._miss_color = [0, 1, 0]
        self._ray_from = []
        self._ray_to = []
        self._ray_ids = []
        self._setup_rays()

    def space(self) -> gym.Space:
        return gym.spaces.Box(low=self._config.min_range,
                              high=self._config.min_range + self._config.range,
                              dtype=np.float32,
                              shape=(self._config.rays,))

    def observe(self) -> np.ndarray:
        rays = self._config.rays
        min_range = self._config.min_range
        max_range = min_range + self._config.range
        results = self._handle.client.rayTestBatch(self._ray_from,
                                                   self._ray_to,
                                                   0,
                                                   parentObjectUniqueId=self._handle.body_id,
                                                   parentLinkIndex=self._handle.link_index)
        scan = np.full(rays, max_range)
        for i in range(rays):
            hit_fraction = results[i][2]
            scan[i] = self._config.range * hit_fraction
        return scan

    def _setup_rays(self):
        rays = self._config.rays
        min_range = self._config.min_range
        max_range = min_range + self._config.range
        for i in range(rays):
            self._ray_from.append([
                min_range * math.sin(-0.5 * 0.25 * 2. * math.pi + 0.75 * 2. * math.pi * float(i) / rays),
                min_range * math.cos(-0.5 * 0.25 * 2. * math.pi + 0.75 * 2. * math.pi * float(i) / rays),
                0
            ])

            self._ray_to.append([
                max_range * math.sin(-0.5 * 0.25 * 2. * math.pi + 0.75 * 2. * math.pi * float(i) / rays),
                max_range * math.cos(-0.5 * 0.25 * 2. * math.pi + 0.75 * 2. * math.pi * float(i) / rays),
                0
            ])


class RGBCamera(SimulatedSensor):

    @dataclass
    class Config:
        width: int
        height: int
        fov: int
        distance: float

    def __init__(self, handle: SimulationHandle, config: Config):
        super().__init__(handle)
        self._config = config

    def space(self) -> gym.Space:
        return gym.spaces.Box(low=0,
                              high=255,
                              shape=(self._config.height, self._config.width, 3),
                              dtype=np.uint8)

    def observe(self) -> np.ndarray:
        state = self._handle.client.getLinkState(bodyUniqueId=self._handle.body_id,
                                                 linkIndex=self._handle.link_index,
                                                 computeForwardKinematics=True)
        position, orientation = state[0], state[1]
        position = (position[0], position[1], position[2])
        rot_matrix = self._handle.client.getMatrixFromQuaternion(orientation)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)
        init_camera_vector = (1, 0, 0)
        init_up_vector = (0, 0, 1)

        camera_vector = rot_matrix.dot(init_camera_vector)
        up_vector = rot_matrix.dot(init_up_vector)
        view_matrix = self._handle.client.computeViewMatrix(position, position + self._config.distance * camera_vector,
                                                          up_vector)
        aspect_ratio = float(self._config.width) / self._config.height

        nearplane, farplane = 0.01, 100
        proj_matrix = self._handle.client.computeProjectionMatrixFOV(self._config.fov, aspect_ratio, nearplane, farplane)
        (_, _, px, _, _) = self._handle.client.getCameraImage(width=self._config.width,
                                                            height=self._config.height,
                                                            renderer=self._handle.client.ER_BULLET_HARDWARE_OPENGL,
                                                            viewMatrix=view_matrix,
                                                            projectionMatrix=proj_matrix)
        rgb_array = np.reshape(px, (self._config.height, self._config.width, -1))
        rgb_array = rgb_array[:, :, :3]
        return rgb_array


class InertialMeasurementUnit(SimulatedSensor):

    @dataclass
    class Config:
        max_acceleration: float
        max_angular_velocity: float

    def __init__(self, handle: SimulationHandle, config: Config):
        super().__init__(handle)
        self._config = config
        self._last_velocity = self._get_velocity()

    def space(self) -> gym.Space:
        high = np.array(3 * [self._config.max_acceleration] + 3 * [self._config.max_angular_velocity])
        low = -high
        return gym.spaces.Box(low=low, high=high)

    def observe(self) -> np.ndarray:
        velocity = self._get_velocity()
        linear_acceleration = (velocity[:3] - self._last_velocity[:3]) / 0.01
        self._last_velocity = velocity
        return np.append(linear_acceleration, velocity[3:])

    def _get_velocity(self):
        v_linear, v_rotation = self._handle.client.getBaseVelocity(bodyUniqueId=self._handle.body_id)
        return np.append(v_linear, v_rotation)

class Tachometer(SimulatedSensor):

    @dataclass
    class Config:
        max_linear_velocity: float
        max_angular_velocity: float

    def __init__(self, handle: SimulationHandle, config: Config):
        super().__init__(handle)
        self._config = config

    def space(self) -> gym.Space:
        high = np.array(3 * [self._config.max_linear_velocity] + 3 * [self._config.max_angular_velocity])
        low = -high
        return gym.spaces.Box(low=low, high=high)

    def observe(self) -> np.ndarray:
        linear, angular =  self._handle.client.getBaseVelocity(bodyUniqueId=self._handle.body_id)
        return np.append(linear, angular)

class GPS(SimulatedSensor):

    @dataclass
    class Config:
        max_x: float
        max_y: float
        max_z: float

    def __init__(self, handle: SimulationHandle, config: Config):
        super().__init__(handle)
        self._config = config

    def space(self) -> gym.Space:
        high = np.array([self._config.max_x, self._config.max_y, self._config.max_z] + 3 * [np.pi])
        low = -high
        return gym.spaces.Box(low=low, high=high)

    def observe(self) -> np.ndarray:
        position, orientation = self._handle.client.getBasePositionAndOrientation(bodyUniqueId=self._handle.body_id)
        orientation = self._handle.client.getEulerFromQuaternion(orientation)
        return np.append(position, orientation)
