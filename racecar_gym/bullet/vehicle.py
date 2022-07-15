from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import pybullet

from racecar_gym.bullet.actuators import BulletActuator
from racecar_gym.bullet.sensors import BulletSensor
from racecar_gym.core.definitions import Pose
from racecar_gym.core.vehicles import Vehicle


class RaceCar(Vehicle):
    @dataclass
    class Config:
        urdf_file: str
        color: Tuple[float, float, float, float]

    def __init__(self, sensors: List[BulletSensor], actuators: List[BulletActuator], config: Config):
        super().__init__()
        self._id = None
        self._config = config
        self._on_finish = False

        self._sensor_indices = {
            'lidar': 4,
            'rgb_camera': 5
        }

        self._actuator_indices = {
            'motor': [8, 15],
            'speed': [8, 15],
            'steering': [0, 2]
        }
        self._actuators = dict([(a.name, a) for a in actuators])
        self._sensors = sensors

    @property
    def id(self) -> Any:
        return self._id

    @property
    def sensors(self) -> List[BulletSensor]:
        return self._sensors

    @property
    def actuators(self) -> Dict[str, BulletActuator]:
        return self._actuators

    def reset(self, pose: Pose):
        if not self._id:
            self._id = self._load_model(self._config.urdf_file, initial_pose=pose)
            self._setup_constraints()
        else:
            pos, orn = pose
            pybullet.resetBasePositionAndOrientation(self._id, pos, pybullet.getQuaternionFromEuler(orn))

        for sensor in self.sensors:
            joint_index = None
            if sensor.type in self._sensor_indices:
                joint_index = self._sensor_indices[sensor.type]
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
        pybullet.changeVisualShape(id, -1, rgbaColor=self._config.color)
        return id

    def _setup_constraints(self):
        car = self._id
        for wheel in range(pybullet.getNumJoints(car)):
            pybullet.setJointMotorControl2(car,
                                          wheel,
                                          pybullet.VELOCITY_CONTROL,
                                          targetVelocity=0,
                                          force=0)
            pybullet.getJointInfo(car, wheel)

            # pybullet.setJointMotorControl2(car,10,pybullet.VELOCITY_CONTROL,targetVelocity=1,force=10)
        c = pybullet.createConstraint(car,
                                     9,
                                     car,
                                     11,
                                     jointType=pybullet.JOINT_GEAR,
                                     jointAxis=[0, 1, 0],
                                     parentFramePosition=[0, 0, 0],
                                     childFramePosition=[0, 0, 0])
        pybullet.changeConstraint(c, gearRatio=1, maxForce=10000)

        c = pybullet.createConstraint(car,
                                     10,
                                     car,
                                     13,
                                     jointType=pybullet.JOINT_GEAR,
                                     jointAxis=[0, 1, 0],
                                     parentFramePosition=[0, 0, 0],
                                     childFramePosition=[0, 0, 0])
        pybullet.changeConstraint(c, gearRatio=-1, maxForce=10000)

        c = pybullet.createConstraint(car,
                                     9,
                                     car,
                                     13,
                                     jointType=pybullet.JOINT_GEAR,
                                     jointAxis=[0, 1, 0],
                                     parentFramePosition=[0, 0, 0],
                                     childFramePosition=[0, 0, 0])
        pybullet.changeConstraint(c, gearRatio=-1, maxForce=10000)

        c = pybullet.createConstraint(car,
                                     16,
                                     car,
                                     18,
                                     jointType=pybullet.JOINT_GEAR,
                                     jointAxis=[0, 1, 0],
                                     parentFramePosition=[0, 0, 0],
                                     childFramePosition=[0, 0, 0])
        pybullet.changeConstraint(c, gearRatio=1, maxForce=10000)

        c = pybullet.createConstraint(car,
                                     16,
                                     car,
                                     19,
                                     jointType=pybullet.JOINT_GEAR,
                                     jointAxis=[0, 1, 0],
                                     parentFramePosition=[0, 0, 0],
                                     childFramePosition=[0, 0, 0])
        pybullet.changeConstraint(c, gearRatio=-1, maxForce=10000)

        c = pybullet.createConstraint(car,
                                     17,
                                     car,
                                     19,
                                     jointType=pybullet.JOINT_GEAR,
                                     jointAxis=[0, 1, 0],
                                     parentFramePosition=[0, 0, 0],
                                     childFramePosition=[0, 0, 0])
        pybullet.changeConstraint(c, gearRatio=-1, maxForce=10000)

        c = pybullet.createConstraint(car,
                                     1,
                                     car,
                                     18,
                                     jointType=pybullet.JOINT_GEAR,
                                     jointAxis=[0, 1, 0],
                                     parentFramePosition=[0, 0, 0],
                                     childFramePosition=[0, 0, 0])
        pybullet.changeConstraint(c, gearRatio=-1, gearAuxLink=15, maxForce=10000)
        c = pybullet.createConstraint(car,
                                     3,
                                     car,
                                     19,
                                     jointType=pybullet.JOINT_GEAR,
                                     jointAxis=[0, 1, 0],
                                     parentFramePosition=[0, 0, 0],
                                     childFramePosition=[0, 0, 0])
        pybullet.changeConstraint(c, gearRatio=-1, gearAuxLink=15, maxForce=10000)