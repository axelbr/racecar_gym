from dataclasses import dataclass
from typing import List, Dict

import pybullet

from racecar_gym.bullet.actuators import BulletActuator
from racecar_gym.bullet.sensors import BulletSensor
from racecar_gym.core.definitions import Pose
from racecar_gym.core.vehicles import Vehicle


class RaceCar(Vehicle):
    @dataclass
    class Config:
        urdf_file: str

    def __init__(self, sensors: List[BulletSensor], actuators: List[BulletActuator], config: Config):
        super().__init__()
        self._id = None
        self._config = config
        self._on_finish = False

        self._sensor_indices = {
            'lidar': 8,
            'rgb_camera': 9
        }

        self._actuator_indices = {
            'motor': [2, 3],
            'steering': [4, 6]
        }
        self._actuators = dict([(a.name, a) for a in actuators])
        self._sensors = sensors

    @property
    def sensors(self) -> List[BulletSensor]:
        return self._sensors

    @property
    def actuators(self) -> Dict[str, BulletActuator]:
        return self._actuators

    def reset(self, pose: Pose):
        self._id = self._load_model(self._config.urdf_file, initial_pose=pose)
        self._setup_constraints()
        for sensor in self.sensors:
            joint_index = None
            if sensor.name in self._sensor_indices:
                joint_index = self._sensor_indices[sensor.name]
            sensor.reset(body_id=self._id, joint_index=joint_index)

        for name, actuator in self.actuators.items():
            joint_indices = None
            if name in self._actuator_indices:
                joint_indices = self._actuator_indices[name]
            actuator.reset(body_id=self._id, joint_indices=joint_indices)

    def _load_model(self, model: str, initial_pose: Pose) -> int:
        position, orientation = initial_pose
        orientation = pybullet.getQuaternionFromEuler(orientation)
        id = pybullet.loadURDF(model, position, orientation)
        return id

    def _setup_constraints(self):
        for wheel in range(pybullet.getNumJoints(self._id)):
            pybullet.setJointMotorControl2(self._id, wheel, pybullet.VELOCITY_CONTROL, targetVelocity=0,
                                           force=0)
            # print(pybullet.getJointInfo(self._id, wheel))
        inactive_wheels = [5, 7]
        for wheel in inactive_wheels:
            pybullet.setJointMotorControl2(self._id, wheel, pybullet.VELOCITY_CONTROL, targetVelocity=0, force=0)
