import math
from dataclasses import dataclass
from time import time
from typing import Dict, Tuple, List, Any

from gym import spaces
import numpy as np
from pybullet_utils.bullet_client import BulletClient

from racecar_gym.models import Pose
from racecar_gym.models.lidar import Lidar


@dataclass
class RaceCarConfig:
    urdf_file: str # path to a urdf file describing the race car
    starting_pose: Pose # initial pose of the car
    motorized_joints: List[str] # names of joints which are used for accelerating the car
    steering_joints: List[str] # names of joints which are used to steer the car
    lidar_joint: str # name of the lidar mounting joint
    camera_joint: str # name of the camera mounting joint
    debug: bool = False # debug flag
    speed_multiplier: float = 20.0
    steering_multiplier: float = 0.5


class RaceCar:

    def __init__(self, client: BulletClient, config: RaceCarConfig):
        """
        Initialize a representation of a race car in the simulation.
        Args:
            client: pybullet client.
            config: race car config instance
        """
        self._client = client
        self._config = config
        self._id = self._load_model(config.urdf_file, config.starting_pose)
        self._joint_dict = self._load_joint_indices(config)
        self._lidar = Lidar(client=self._client,
                            id=self._joint_dict[config.lidar_joint],
                            car_id=self._id,
                            rays=100,
                            rendering=config.debug,
                            min_range=0.25,
                            max_range=5.0)

        self._setup_constraints()
        self._on_finish = False
        self._lap = 0

    def _load_model(self, model: str, initial_pose: Pose) -> int:
        position, orientation = initial_pose
        orientation = self._client.getQuaternionFromEuler(orientation)
        id = self._client.loadURDF(model, position, orientation)
        return id

    def _setup_constraints(self):

        for wheel in range(self._client.getNumJoints(self._id)):
            print("joint[", wheel, "]=", self._client.getJointInfo(self._id, wheel))
            self._client.setJointMotorControl2(self._id, wheel, self._client.VELOCITY_CONTROL, targetVelocity=0,
                                               force=0)
            self._client.getJointInfo(self._id, wheel)

        # self._client.setJointMotorControl2(self._id,10,self._client.VELOCITY_CONTROL,targetVelocity=1,force=10)
        c = self._client.createConstraint(self._id, 9, self._id, 11, jointType=self._client.JOINT_GEAR,
                                          jointAxis=[0, 1, 0],
                                          parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
        self._client.changeConstraint(c, gearRatio=1, maxForce=10000)

        c = self._client.createConstraint(self._id, 10, self._id, 13, jointType=self._client.JOINT_GEAR,
                                          jointAxis=[0, 1, 0],
                                          parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
        self._client.changeConstraint(c, gearRatio=-1, maxForce=10000)

        c = self._client.createConstraint(self._id, 9, self._id, 13, jointType=self._client.JOINT_GEAR,
                                          jointAxis=[0, 1, 0],
                                          parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
        self._client.changeConstraint(c, gearRatio=-1, maxForce=10000)

        c = self._client.createConstraint(self._id, 16, self._id, 18, jointType=self._client.JOINT_GEAR,
                                          jointAxis=[0, 1, 0],
                                          parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
        self._client.changeConstraint(c, gearRatio=1, maxForce=10000)

        c = self._client.createConstraint(self._id, 16, self._id, 19, jointType=self._client.JOINT_GEAR,
                                          jointAxis=[0, 1, 0],
                                          parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
        self._client.changeConstraint(c, gearRatio=-1, maxForce=10000)

        c = self._client.createConstraint(self._id, 17, self._id, 19, jointType=self._client.JOINT_GEAR,
                                          jointAxis=[0, 1, 0],
                                          parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
        self._client.changeConstraint(c, gearRatio=-1, maxForce=10000)

        c = self._client.createConstraint(self._id, 1, self._id, 18, jointType=self._client.JOINT_GEAR,
                                          jointAxis=[0, 1, 0],
                                          parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
        self._client.changeConstraint(c, gearRatio=-1, gearAuxLink=15, maxForce=10000)
        c = self._client.createConstraint(self._id, 3, self._id, 19, jointType=self._client.JOINT_GEAR,
                                          jointAxis=[0, 1, 0],
                                          parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
        self._client.changeConstraint(c, gearRatio=-1, gearAuxLink=15, maxForce=10000)

    @property
    def id(self) -> int:
        """
        Get the id of the object in the current simulation.
        Returns:
            id
        """
        return self._id

    @property
    def lap(self):
        return self._lap

    @lap.setter
    def lap(self, value):
        self._lap = value

    @property
    def on_finish(self) -> bool:
        return self._on_finish

    @on_finish.setter
    def on_finish(self, value):
        self._on_finish = value

    def step(self, velocity: float, steering_angle: float, force: float) -> None:
        motorized = [self._joint_dict[joint] for joint in self._config.motorized_joints]
        steering = [self._joint_dict[joint] for joint in self._config.steering_joints]

        for wheel in motorized:
            self._client.setJointMotorControl2(self._id, wheel, self._client.VELOCITY_CONTROL,
                                               targetVelocity=velocity * self._config.speed_multiplier, force=force)

        for steer in steering:
            self._client.setJointMotorControl2(self._id, steer, self._client.POSITION_CONTROL,
                                               targetPosition=-steering_angle * self._config.steering_multiplier)

    def observe(self, sensors: List[str]) -> Dict[str, np.ndarray]:
        observations = {}
        for sensor in sensors:
            if sensor == 'odometry':
                observations['pose'], observations['velocity'] = self._odometry()
            if sensor == 'lidar':
                observations[sensor] = self._lidar_scan()
        observations['lap'] = self.lap
        return observations

    def status(self) -> Dict[str, Any]:
        state = {}
        state['collisions'] = set([c[2] for c in self._client.getContactPoints(self._id)])
        return state

    def _odometry(self) -> Tuple[np.ndarray, np.ndarray]:
        position, orientation = self._client.getBasePositionAndOrientation(self._id)
        euler = self._client.getEulerFromQuaternion(orientation)
        pose = np.append(position, euler)
        velocities = self._client.getBaseVelocity(self._id)
        velocities = np.array(velocities[0] + velocities[1])
        return pose, velocities

    def _lidar_scan(self) -> np.ndarray:
        return self._lidar.scan()

    def reset(self):
        self._id = self._load_model(self._config.urdf_file, self._config.starting_pose)
        self._setup_constraints()
        self._lidar.reset()

    def _load_joint_indices(self, config: RaceCarConfig) -> Dict[str, int]:
        available_joints = config.motorized_joints \
                           + config.steering_joints \
                           + [config.lidar_joint, config.camera_joint]

        joint_dict = {}

        for joint_index in range(self._client.getNumJoints(self._id)):
            joint_name = self._client.getJointInfo(self._id, joint_index)[1].decode('UTF-8')
            if joint_name in available_joints:
                joint_dict[joint_name] = joint_index

        return joint_dict
