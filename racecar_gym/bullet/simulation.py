from dataclasses import dataclass

import numpy as np
from nptyping import NDArray
import pybullet as p

from racecar_gym.models.definitions import Pose, Velocity


@dataclass
class SimulationHandle:
    link_index: int
    body_id: int


def get_pose(id: int, index: int = None) -> Pose:
    if not index:
        position, orientation = p.getBasePositionAndOrientation(id)
    else:
        state = p.getLinkState(id, linkIndex=index, computeForwardKinematics=True)
        position, orientation = state[0], state[1]
    orientation = p.getEulerFromQuaternion(orientation)
    return position, orientation


def get_velocity(id: int) -> Velocity:
    v_linear, v_rotation = p.getBaseVelocity(id)
    return v_linear + v_rotation

class Camera:

    def __init__(self, target_distance: float, fov: float, near_plane: float = 0.01, far_plane: float = 100):
        self._position = np.zeros(3)
        self._orientation = np.zeros(4)
        self._up_vector =[0, 0, 1]
        self._camera_vector = [1, 0, 0]
        self._target_distance = target_distance
        self._fov = fov
        self._near_plane = near_plane
        self._far_plane = far_plane

    def set_pose(self, pose: Pose):
        position, orientation = pose
        self._position = np.array(position)
        self._orientation = np.array(p.getQuaternionFromEuler(orientation))

    def get_image(self, height: int, width: int) -> NDArray:
        rot_matrix = p.getMatrixFromQuaternion(self._orientation)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)
        camera_vector = rot_matrix.dot(self._camera_vector)
        up_vector = rot_matrix.dot(self._up_vector)
        target = self._position + self._target_distance * camera_vector
        view_matrix = p.computeViewMatrix(self._position, target, up_vector)
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

class RayCast:

    def __init__(self, rays: int, min_distance: float, scan_range: float):
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

        self._from = np.array(from_points)
        self._to = np.array(to_points)
        self._rays = rays
        self._range = scan_range
        self._hit_color = [1, 0, 0]
        self._miss_color = [0, 1, 0]
        self._ray_ids = []

    def scan_from_link(self, body_id: int, index: int, debug: bool = False):
        results = p.rayTestBatch(self._from, self._to, 0, parentObjectUniqueId=body_id, parentLinkIndex=index)
        scan = np.full(self._rays, self._range)

        for i in range(self._rays):
            hit_fraction = results[i][2]
            scan[i] = self._range * hit_fraction

            if debug:
                if len(self._ray_ids) < self._rays:
                    ray_id = p.addUserDebugLine(self._from[i], self._to[i], self._miss_color,
                                                parentObjectUniqueId=body_id,
                                                parentLinkIndex=index)
                    self._ray_ids.append(ray_id)

                if (hit_fraction == 1.):
                    p.addUserDebugLine(self._from[i], self._to[i], self._miss_color,
                                                  replaceItemUniqueId=self._ray_ids[i],
                                                  parentObjectUniqueId=body_id,
                                                  parentLinkIndex=index)
                else:
                    localHitTo = [
                        self._from[i][0] + hit_fraction * (self._to[i][0] - self._from[i][0]),
                        self._from[i][1] + hit_fraction * (self._to[i][1] - self._from[i][1]),
                        self._from[i][2] + hit_fraction * (self._to[i][2] - self._from[i][2])]

                    p.addUserDebugLine(self._from[i],
                                       localHitTo,
                                       self._hit_color,
                                       replaceItemUniqueId=self._ray_ids[i],
                                       parentObjectUniqueId=body_id, parentLinkIndex=index)
        return scan