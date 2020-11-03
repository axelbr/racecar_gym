import numpy as np
import pybullet
from nptyping import NDArray


def get_velocity(id: int) -> NDArray[(6,), np.float]:
    linear, angular = pybullet.getBaseVelocity(id)
    position, orientation = pybullet.getBasePositionAndOrientation(id)
    rotation = pybullet.getMatrixFromQuaternion(orientation)
    rotation = np.reshape(rotation, (-1, 3)).transpose()
    linear = rotation.dot(linear)
    angular = rotation.dot(angular)
    return np.append(linear, angular)


def get_pose(id: int) -> NDArray[(6,), np.float]:
    position, orientation = pybullet.getBasePositionAndOrientation(id)
    orientation = pybullet.getEulerFromQuaternion(orientation)
    pose = np.append(position, orientation)
    return pose
