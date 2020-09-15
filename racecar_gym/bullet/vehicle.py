from typing import Dict, Tuple, List, Any, Optional, Set

import gym
import numpy as np
import pybullet
from pybullet_utils.bullet_client import BulletClient

from racecar_gym.bullet.sensors import Lidar, RGBCamera, InertialMeasurementUnit, Tachometer, GPS
from racecar_gym.bullet.simulation import SimulationHandle
from racecar_gym.models.definitions import Pose
from racecar_gym.models.map import Map
from racecar_gym.bullet.configs import VehicleConfig, VehicleJointConfig, SensorConfig
from racecar_gym.models.sensors import Sensor, FixedTimestepSensor

from racecar_gym.models.vehicles import Vehicle


class RaceCarDifferential(Vehicle):

    def space(self, sensors: List[str]) -> gym.spaces.Dict:
        observation_space =  gym.spaces.Dict()
        for sensor in sensors:
            observation_space.spaces[sensor] = self._sensors[sensor].space()
        return observation_space

    def __init__(self, map: Map, config: VehicleConfig, position: int):
        self._car_id = position
        #pybullet = BulletClient(pybullet.SHARED_MEMORY)
        self._config = config
        self._id = None

        self._joint_dict = {}
        self._on_finish = False
        self._lap = 0
        self._map = map
        self._initial_pose = self._map.starting_pose(position)
        self._sensors = {}
        self._action_space = gym.spaces.Box(
            low=np.array([-config.max_speed, -config.max_steering_angle, 0]),
            high=np.array([config.max_speed, config.max_steering_angle, config.max_force]),
        )

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self) -> gym.Space:
        space = gym.spaces.Dict()
        for sensor_type, sensor in self._sensors.items():
            space.spaces[sensor_type] = sensor.space()
        return space

    def _create_sensor(self, config: SensorConfig) -> Sensor:
        link_index = None
        if config.link:
            link_index = self._joint_dict[config.link]
        handle = SimulationHandle(link_index=link_index, body_id=self.id)
        sensor = None
        if config.type == 'lidar':
            sensor = Lidar(handle=handle, config=Lidar.Config(**config.params))
        elif config.type == 'rgb_camera':
            sensor = RGBCamera(handle=handle, config=RGBCamera.Config(**config.params))
        elif config.type == 'imu':
            sensor = InertialMeasurementUnit(handle=handle, config=InertialMeasurementUnit.Config(**config.params))
        elif config.type == 'gps':
            sensor = GPS(handle=handle, config=GPS.Config(**config.params))
        elif config.type == 'tacho':
            sensor = Tachometer(handle=handle, config=Tachometer.Config(**config.params))
        else:
            NotImplementedError('No such sensor type implemented')
        timestep = pybullet.getPhysicsEngineParameters()['fixedTimeStep']
        return FixedTimestepSensor(sensor=sensor, frequency=config.frequency, time_step=timestep)

    @property
    def id(self) -> int:
        """
        Get the id of the object in the current simulation.
        Returns:
            id
        """
        return self._id

    @property
    def config(self) -> VehicleConfig:
        return self._config


    def control(self, command) -> None:
        velocity = command['velocity']
        steering_angle = command['steering_angle']
        force = command['force']
        joints = self._config.joints
        motorized = [self._joint_dict[joint] for joint in joints.motorized_joints]
        steering = [self._joint_dict[joint] for joint in joints.steering_joints]

        for wheel in motorized:
            pybullet.setJointMotorControl2(self._id, wheel, pybullet.VELOCITY_CONTROL,
                                               targetVelocity=velocity * self._config.speed_multiplier, force=force)

        for steer in steering:
            pybullet.setJointMotorControl2(self._id, steer, pybullet.POSITION_CONTROL,
                                               targetPosition=-steering_angle * self._config.steering_multiplier)

        self._update_lap()

    def query(self, sensors: List[str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        observations = {}
        info = {}
        for sensor_type in sensors:
            if sensor_type in self._sensors.keys():
                observations[sensor_type] = self._sensors[sensor_type].observe()
        observations['lap'] = self._lap
        info['collisions'] = self._check_collisions()
        observations['collision'] = len(info['collisions']) > 0
        return observations, info

    def reset(self):
        self._id = self._load_model(self._config.urdf_file, initial_pose=self._initial_pose)
        self._joint_dict = self._load_joint_indices(self._config.joints)
        self._setup_constraints()
        self._sensors = {}
        for sensor_config in self._config.sensors:
            self._sensors[sensor_config.type] = self._create_sensor(sensor_config)



    def _load_joint_indices(self, config: VehicleJointConfig) -> Dict[str, int]:
        available_joints = config.motorized_joints \
                           + config.steering_joints \
                           + config.lidar_joint \
                           + config.camera_joint

        joint_dict = {}

        for joint_index in range(pybullet.getNumJoints(self._id)):
            joint_name = pybullet.getJointInfo(self._id, joint_index)[1].decode('UTF-8')
            if joint_name in available_joints:
                joint_dict[joint_name] = joint_index

        return joint_dict

    def _load_model(self, model: str, initial_pose: Pose) -> int:
        position, orientation = initial_pose
        orientation = pybullet.getQuaternionFromEuler(orientation)
        id = pybullet.loadURDF(model, position, orientation)
        return id

    def _setup_constraints(self):

        for wheel in range(pybullet.getNumJoints(self._id)):
            pybullet.setJointMotorControl2(self._id, wheel, pybullet.VELOCITY_CONTROL, targetVelocity=0,
                                               force=0)
            pybullet.getJointInfo(self._id, wheel)

        # pybullet.setJointMotorControl2(self._id,10,pybullet.VELOCITY_CONTROL,targetVelocity=1,force=10)
        c = pybullet.createConstraint(self._id, 9, self._id, 11, jointType=pybullet.JOINT_GEAR,
                                          jointAxis=[0, 1, 0],
                                          parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
        pybullet.changeConstraint(c, gearRatio=1, maxForce=10000)

        c = pybullet.createConstraint(self._id, 10, self._id, 13, jointType=pybullet.JOINT_GEAR,
                                          jointAxis=[0, 1, 0],
                                          parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
        pybullet.changeConstraint(c, gearRatio=-1, maxForce=10000)

        c = pybullet.createConstraint(self._id, 9, self._id, 13, jointType=pybullet.JOINT_GEAR,
                                          jointAxis=[0, 1, 0],
                                          parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
        pybullet.changeConstraint(c, gearRatio=-1, maxForce=10000)

        c = pybullet.createConstraint(self._id, 16, self._id, 18, jointType=pybullet.JOINT_GEAR,
                                          jointAxis=[0, 1, 0],
                                          parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
        pybullet.changeConstraint(c, gearRatio=1, maxForce=10000)

        c = pybullet.createConstraint(self._id, 16, self._id, 19, jointType=pybullet.JOINT_GEAR,
                                          jointAxis=[0, 1, 0],
                                          parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
        pybullet.changeConstraint(c, gearRatio=-1, maxForce=10000)

        c = pybullet.createConstraint(self._id, 17, self._id, 19, jointType=pybullet.JOINT_GEAR,
                                          jointAxis=[0, 1, 0],
                                          parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
        pybullet.changeConstraint(c, gearRatio=-1, maxForce=10000)

        c = pybullet.createConstraint(self._id, 1, self._id, 18, jointType=pybullet.JOINT_GEAR,
                                          jointAxis=[0, 1, 0],
                                          parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
        pybullet.changeConstraint(c, gearRatio=-1, gearAuxLink=15, maxForce=10000)
        c = pybullet.createConstraint(self._id, 3, self._id, 19, jointType=pybullet.JOINT_GEAR,
                                          jointAxis=[0, 1, 0],
                                          parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
        pybullet.changeConstraint(c, gearRatio=-1, gearAuxLink=15, maxForce=10000)

    def _update_lap(self):
        closest_points = pybullet.getClosestPoints(self._id, self._map.finish_id, 0.05)
        if len(closest_points) > 0:
            if not self._on_finish:
                self._on_finish = True
                self._lap += 1
        else:
            if self._on_finish:
                self._on_finish = False

    def _check_collisions(self) -> Set[int]:
        collisions = set([c[2] for c in pybullet.getContactPoints(self._id)])
        collisions_without_floor = collisions - {self._map.floor_id, self._map.finish_id}
        return collisions_without_floor
