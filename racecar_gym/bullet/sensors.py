from abc import ABC
from dataclasses import dataclass
from typing import Any, TypeVar

import gym
import numpy as np
import pybullet as p
from nptyping import NDArray

from racecar_gym.bullet.world import World
from racecar_gym.core import Sensor

T = TypeVar('T')


class BulletSensor(Sensor[T], ABC):

    def __init__(self, name: str, type: str):
        super().__init__(name, type)
        self._body_id = None
        self._joint_index = None

    def reset(self, body_id: int, joint_index: int = None):
        self._body_id = body_id
        self._joint_index = joint_index

    @property
    def body_id(self) -> int:
        return self._body_id

    @property
    def joint_index(self) -> int:
        return self._joint_index


class FixedTimestepSensor(BulletSensor[T], ABC):

    def __init__(self, sensor: BulletSensor, frequency: float, time_step: float):
        super().__init__(sensor.name, sensor.type)
        self._sensor = sensor
        self._frequency = 1.0 / frequency
        self._time_step = time_step
        self._last_timestep = 0
        self._last_observation = None

    def space(self) -> gym.Space:
        return self._sensor.space()

    def observe(self) -> T:
        self._last_timestep += self._time_step
        if self._last_timestep >= self._frequency or self._last_observation is None:
            self._last_observation = self._sensor.observe()
            self._last_timestep = 0
        return self._last_observation

    def reset(self, body_id: int, joint_index: int = None):
        self._sensor.reset(body_id=body_id, joint_index=joint_index)


class Lidar(BulletSensor[NDArray[(Any,), np.float]]):
    @dataclass
    class Config:
        rays: int
        range: float
        min_range: float
        debug: bool = False

    def __init__(self, name: str, type: str, config: Config):
        super().__init__(name, type)
        self._config = config
        self._min_range = config.min_range
        self._rays = self._config.rays
        self._range = self._config.range
        self._hit_color = [1, 0, 0]
        self._miss_color = [0, 1, 0]
        self._ray_ids = []

        self._from, self._to = self._setup_raycast(min_distance=self._min_range,
                                                   scan_range=self._range,
                                                   rays=self._rays)

    def _setup_raycast(self, min_distance: float, scan_range: float, rays: int):
        start = min_distance
        end = min_distance + scan_range
        from_points, to_points = [], []
        for i in range(rays):
            from_points.append([
                start * np.sin(-0.5 * 0.25 * 2. * np.pi + 0.75 * 2. * np.pi * float(i) / rays),
                start * np.cos(-0.5 * 0.25 * 2. * np.pi + 0.75 * 2. * np.pi * float(i) / rays),
                0
            ])

            to_points.append([
                end * np.sin(-0.5 * 0.25 * 2. * np.pi + 0.75 * 2. * np.pi * float(i) / rays),
                end * np.cos(-0.5 * 0.25 * 2. * np.pi + 0.75 * 2. * np.pi * float(i) / rays),
                0
            ])

        return np.array(from_points), np.array(to_points)

    def space(self) -> gym.Space:
        return gym.spaces.Box(low=self._min_range,
                              high=self._min_range + self._range,
                              dtype=np.float32,
                              shape=(self._rays,))

    def observe(self) -> NDArray[(Any,), np.float]:
        results = p.rayTestBatch(self._from, self._to, 0,
                                 parentObjectUniqueId=self.body_id,
                                 parentLinkIndex=self.joint_index)
        scan = np.full(self._rays, self._range)

        for i in range(self._rays):
            hit_fraction = results[i][2]
            scan[i] = self._range * hit_fraction

            if self._config.debug:
                if len(self._ray_ids) < self._rays:
                    ray_id = p.addUserDebugLine(self._from[i], self._to[i], self._miss_color,
                                                parentObjectUniqueId=self.body_id,
                                                parentLinkIndex=self.joint_index)
                    self._ray_ids.append(ray_id)

                if (hit_fraction == 1.):
                    p.addUserDebugLine(self._from[i], self._to[i], self._miss_color,
                                       replaceItemUniqueId=self._ray_ids[i],
                                       parentObjectUniqueId=self.body_id,
                                       parentLinkIndex=self.joint_index)
                else:
                    localHitTo = [
                        self._from[i][0] + hit_fraction * (self._to[i][0] - self._from[i][0]),
                        self._from[i][1] + hit_fraction * (self._to[i][1] - self._from[i][1]),
                        self._from[i][2] + hit_fraction * (self._to[i][2] - self._from[i][2])]

                    p.addUserDebugLine(self._from[i],
                                       localHitTo,
                                       self._hit_color,
                                       replaceItemUniqueId=self._ray_ids[i],
                                       parentObjectUniqueId=self.body_id,
                                       parentLinkIndex=self.joint_index)
        return scan


class RGBCamera(BulletSensor[NDArray[(Any, Any, 3), np.int]]):
    @dataclass
    class Config:
        width: int
        height: int
        fov: int
        distance: float
        near_plane: float
        far_plane: float

    def __init__(self, name: str, type: str, config: Config):
        super().__init__(name, type)
        self._config = config
        self._up_vector = [0, 0, 1]
        self._camera_vector = [1, 0, 0]
        self._target_distance = config.distance
        self._fov = config.fov
        self._near_plane = config.near_plane
        self._far_plane = config.far_plane

    def space(self) -> gym.Space:
        return gym.spaces.Box(low=0,
                              high=255,
                              shape=(self._config.height, self._config.width, 3),
                              dtype=np.uint8)

    def observe(self) -> NDArray[(Any, Any, 3), np.int]:
        width, height = self._config.width, self._config.height
        state = p.getLinkState(self.body_id, linkIndex=self.joint_index, computeForwardKinematics=True)
        position, orientation = state[0], state[1]
        rot_matrix = p.getMatrixFromQuaternion(orientation)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)
        camera_vector = rot_matrix.dot(self._camera_vector)
        up_vector = rot_matrix.dot(self._up_vector)
        target = position + self._target_distance * camera_vector
        view_matrix = p.computeViewMatrix(position, target, up_vector)
        aspect_ratio = float(width) / height
        proj_matrix = p.computeProjectionMatrixFOV(self._fov, aspect_ratio, self._near_plane, self._far_plane)
        (_, _, px, _, _) = p.getCameraImage(width=width,
                                            height=height,
                                            renderer=p.ER_BULLET_HARDWARE_OPENGL,
                                            viewMatrix=view_matrix,
                                            projectionMatrix=proj_matrix)

        rgb_array = np.reshape(px, (height, width, -1))
        rgb_array = rgb_array[:, :, :3]
        return rgb_array


class IMU(BulletSensor[NDArray[(6,), np.float]]):
    @dataclass
    class Config:
        max_acceleration: float
        max_angular_velocity: float

    def __init__(self, name: str, type: str, config: Config):
        super().__init__(name, type)
        self._config = config
        self._last_velocity = np.zeros(shape=6)

    def space(self) -> gym.Space:
        high = np.array(3 * [self._config.max_acceleration] + 3 * [self._config.max_angular_velocity])
        low = -high
        return gym.spaces.Box(low=low, high=high)

    def _get_velocity(self):
        v_linear, v_rotation = p.getBaseVelocity(self.body_id)
        return v_linear + v_rotation

    def observe(self) -> NDArray[(6,), np.float]:
        velocity = np.array(self._get_velocity())
        linear_acceleration = (velocity[:3] - self._last_velocity[:3]) / 0.01
        self._last_velocity = velocity
        return np.append(linear_acceleration, velocity[3:])


class Tachometer(BulletSensor[NDArray[(6,), np.float]]):
    @dataclass
    class Config:
        max_linear_velocity: float
        max_angular_velocity: float

    def __init__(self, name: str, type: str, config: Config):
        super().__init__(name, type)
        self._config = config

    def _get_velocity(self):
        v_linear, v_rotation = p.getBaseVelocity(self.body_id)
        return v_linear + v_rotation

    def space(self) -> gym.Space:
        high = np.array(3 * [self._config.max_linear_velocity] + 3 * [self._config.max_angular_velocity])
        low = -high
        return gym.spaces.Box(low=low, high=high)

    def observe(self) -> NDArray[(6,), np.float]:
        velocity = self._get_velocity()
        return np.array(velocity)


class GPS(BulletSensor[NDArray[(6,), np.float]]):
    @dataclass
    class Config:
        max_x: float
        max_y: float
        max_z: float

    def __init__(self, name: str, type: str, config: Config):
        super().__init__(name, type)
        self._config = config

    def space(self) -> gym.Space:
        high = np.array([self._config.max_x, self._config.max_y, self._config.max_z] + 3 * [np.pi])
        low = -high
        return gym.spaces.Box(low=low, high=high)

    def observe(self) -> NDArray[(6,), np.float]:
        position, orientation = p.getBasePositionAndOrientation(self.body_id)
        return np.append(position, orientation)


class LapCounter(BulletSensor[int]):
    @dataclass
    class Config:
        max_laps: int
        margin: float

    def __init__(self, name: str, type: str, config: Config):
        super().__init__(name, type)
        self._config = config
        self._on_finish = False
        self._lap = 0

    def space(self) -> gym.Space:
        return gym.spaces.Discrete(self._config.max_laps)

    def observe(self) -> int:
        closest_points = p.getClosestPoints(self.body_id, World.FINISH_ID, self._config.margin)
        if len(closest_points) > 0:
            if not self._on_finish:
                self._on_finish = True
                self._lap += 1
        else:
            if self._on_finish:
                self._on_finish = False

        return self._lap