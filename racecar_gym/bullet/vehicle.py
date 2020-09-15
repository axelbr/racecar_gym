from dataclasses import dataclass
from typing import List

import pybullet

from racecar_gym.entities.actuators import Actuator
from racecar_gym.entities.definitions import Pose
from racecar_gym.entities.sensors import Sensor
from racecar_gym.entities.vehicles import Vehicle


class RaceCarDifferential(Vehicle):
    @dataclass
    class Config:
        urdf_file: str

    def __init__(self, sensors: List[Sensor], actuators: List[Actuator], config: Config):
        super().__init__(sensors, actuators)
        self._id = None
        self._config = config
        self._on_finish = False

    def reset(self, pose: Pose):
        self._id = self._load_model(self._config.urdf_file, initial_pose=pose)
        self._setup_constraints()

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
