import math
from time import time
from typing import Dict, Tuple, List

from gym import spaces
import numpy as np
from pybullet_utils.bullet_client import BulletClient

class Lidar:

    def __init__(self, rays: int, min_range: float, max_range: float, client: BulletClient, car_id: int, id: int, rendering: bool = False):
        self._rays = rays
        self._client = client
        self._id = id
        self._car_id = car_id
        self._render = rendering
        self._min_range = min_range
        self._max_range = max_range
        self._range = max_range - min_range
        self._hit_color = [1, 0, 0]
        self._miss_color = [0, 1, 0]
        self._ray_from = []
        self._ray_to = []
        self._ray_ids = []
        self._last_scan_time = time()
        self._setup_rays()

    @property
    def last_scan_time(self):
        return self._last_scan_time

    def _setup_rays(self):
        for i in range(self._rays):

            self._ray_from.append([
                self._min_range * math.sin(-0.5 * 0.25 * 2. * math.pi + 0.75 * 2. * math.pi * float(i) / self._rays),
                self._min_range * math.cos(-0.5 * 0.25 * 2. * math.pi + 0.75 * 2. * math.pi * float(i) / self._rays),
                0
            ])

            self._ray_to.append([
                self._max_range * math.sin(-0.5 * 0.25 * 2. * math.pi + 0.75 * 2. * math.pi * float(i) / self._rays),
                self._max_range * math.cos(-0.5 * 0.25 * 2. * math.pi + 0.75 * 2. * math.pi * float(i) / self._rays),
                0
            ])

            if self._render:
                ray_id = self._client.addUserDebugLine(self._ray_from[i], self._ray_to[i], self._miss_color,
                                                       parentObjectUniqueId=self._car_id,
                                                       parentLinkIndex=self._id)
                self._ray_ids.append(ray_id)

        results = self._client.rayTestBatch(self._ray_from, self._ray_to, 0, parentObjectUniqueId=self._car_id, parentLinkIndex=self._id)

        for i in range(self._rays):
            hitFraction = results[i][2]
            if (hitFraction == 1.):
                self._client.addUserDebugLine(self._ray_from[i], self._ray_to[i], self._miss_color, replaceItemUniqueId=self._ray_ids[i],
                                              parentObjectUniqueId=self._car_id, parentLinkIndex=self._id)
            else:
                localHitTo = [self._ray_from[i][0] + hitFraction * (self._ray_to[i][0] - self._ray_from[i][0]),
                              self._ray_from[i][1] + hitFraction * (self._ray_to[i][1] - self._ray_from[i][1]),
                              self._ray_from[i][2] + hitFraction * (self._ray_to[i][2] - self._ray_from[i][2])]
                self._client.addUserDebugLine(self._ray_from[i], localHitTo, self._hit_color, replaceItemUniqueId=self._ray_ids[i],
                                              parentObjectUniqueId=self._car_id, parentLinkIndex=self._id)
        self._last_scan_time = time()

    def _visualize(self, ray: int, hit_fraction: float):
        if (hit_fraction == 1.):
            self._client.addUserDebugLine(self._ray_from[ray], self._ray_to[ray], self._miss_color,
                                          replaceItemUniqueId=self._ray_ids[ray], parentObjectUniqueId=self._car_id,
                                          parentLinkIndex=self._id)
        else:
            localHitTo = [self._ray_from[ray][0] + hit_fraction * (self._ray_to[ray][0] - self._ray_from[ray][0]),
                          self._ray_from[ray][1] + hit_fraction * (self._ray_to[ray][1] - self._ray_from[ray][1]),
                          self._ray_from[ray][2] + hit_fraction * (self._ray_to[ray][2] - self._ray_from[ray][2])]

            self._client.addUserDebugLine(self._ray_from[ray], localHitTo, self._hit_color,
                                          replaceItemUniqueId=self._ray_ids[ray],
                                          parentObjectUniqueId=self._car_id, parentLinkIndex=self._id)

    def scan(self) -> np.ndarray:
        results = self._client.rayTestBatch(self._ray_from, self._ray_to, 0, parentObjectUniqueId=self._car_id, parentLinkIndex=self._id)
        scan = np.full(self._rays, self._max_range)
        for i in range(self._rays):
            hit_fraction = results[i][2]
            scan[i] = self._range * hit_fraction
            if self._render:
                self._visualize(ray=i, hit_fraction=hit_fraction)
        self._last_scan_time = time()
        return scan

    def reset(self):
        self._setup_rays()



class RaceCar:

    def __init__(self, client: BulletClient, initial_pose: np.ndarray, model: str):
        self._client = client
        self._model_path = model
        self._initial_pose = initial_pose
        self._id = self._load_model(model, initial_pose)

        self._motorized_wheel_ids = [8, 15]
        self._steering_link_ids = [0,2]
        self._hokuyo_link_id = 4
        self._speed_multiplier = 20.
        self._steering_multiplier = 0.5
        self._lidar = Lidar(client=self._client,
                            id=self._hokuyo_link_id,
                            car_id=self._id,
                            rays=100,
                            rendering=True,
                            min_range=0.25,
                            max_range=5.0)

        self._setup_constraints()

    def _load_model(self, model: str, initial_pose: np.ndarray) -> int:
        position = [initial_pose[0], initial_pose[1], 0.1]
        orientation = self._client.getQuaternionFromEuler([0, 0, initial_pose[2]])
        return self._client.loadURDF(model, position, orientation)

    def _setup_constraints(self):

        for wheel in range(self._client.getNumJoints(self._id)):
            print("joint[", wheel, "]=", self._client.getJointInfo(self._id, wheel))
            self._client.setJointMotorControl2(self._id, wheel, self._client.VELOCITY_CONTROL, targetVelocity=0, force=0)
            self._client.getJointInfo(self._id, wheel)

        # self._client.setJointMotorControl2(self._id,10,self._client.VELOCITY_CONTROL,targetVelocity=1,force=10)
        c = self._client.createConstraint(self._id, 9, self._id, 11, jointType=self._client.JOINT_GEAR, jointAxis=[0, 1, 0],
                               parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
        self._client.changeConstraint(c, gearRatio=1, maxForce=10000)

        c = self._client.createConstraint(self._id, 10, self._id, 13, jointType=self._client.JOINT_GEAR, jointAxis=[0, 1, 0],
                               parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
        self._client.changeConstraint(c, gearRatio=-1, maxForce=10000)

        c = self._client.createConstraint(self._id, 9, self._id, 13, jointType=self._client.JOINT_GEAR, jointAxis=[0, 1, 0],
                               parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
        self._client.changeConstraint(c, gearRatio=-1, maxForce=10000)

        c = self._client.createConstraint(self._id, 16, self._id, 18, jointType=self._client.JOINT_GEAR, jointAxis=[0, 1, 0],
                               parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
        self._client.changeConstraint(c, gearRatio=1, maxForce=10000)

        c = self._client.createConstraint(self._id, 16, self._id, 19, jointType=self._client.JOINT_GEAR, jointAxis=[0, 1, 0],
                               parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
        self._client.changeConstraint(c, gearRatio=-1, maxForce=10000)

        c = self._client.createConstraint(self._id, 17, self._id, 19, jointType=self._client.JOINT_GEAR, jointAxis=[0, 1, 0],
                               parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
        self._client.changeConstraint(c, gearRatio=-1, maxForce=10000)

        c = self._client.createConstraint(self._id, 1, self._id, 18, jointType=self._client.JOINT_GEAR, jointAxis=[0, 1, 0],
                               parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
        self._client.changeConstraint(c, gearRatio=-1, gearAuxLink=15, maxForce=10000)
        c = self._client.createConstraint(self._id, 3, self._id, 19, jointType=self._client.JOINT_GEAR, jointAxis=[0, 1, 0],
                               parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
        self._client.changeConstraint(c, gearRatio=-1, gearAuxLink=15, maxForce=10000)

    @property
    def id(self) -> int:
        return self._id

    def step(self, velocity: float, steering_angle: float, force: float) -> None:
        for wheel in self._motorized_wheel_ids:
            self._client.setJointMotorControl2(self._id, wheel, self._client.VELOCITY_CONTROL, targetVelocity=velocity*self._speed_multiplier, force=force)

        for steer in self._steering_link_ids:
            self._client.setJointMotorControl2(self._id, steer, self._client.POSITION_CONTROL, targetPosition=-steering_angle*self._steering_multiplier)

    def observe(self, sensors: List[str]) -> Dict[str, np.ndarray]:
        observations = {}

        for sensor in sensors:
            if sensor == 'odometry':
                observations['pose'], observations['velocity'] = self._odometry()
            if sensor == 'lidar':
                observations[sensor] = self._lidar_scan()

        return observations

    def _odometry(self) -> Tuple[np.ndarray, np.ndarray]:
        position, orientation = self._client.getBasePositionAndOrientation(self._id)
        yaw = self._client.getEulerFromQuaternion(orientation)
        pose = np.append(position, yaw)
        velocities = self._client.getBaseVelocity(self._id)
        velocities = np.array(velocities[0] + velocities[1])
        return pose, velocities

    def _lidar_scan(self) -> np.ndarray:
        return self._lidar.scan()

    def reset(self):
        self._id = self._load_model(self._model_path, self._initial_pose)
        self._setup_constraints()
        self._lidar.reset()