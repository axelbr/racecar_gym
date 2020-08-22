from typing import Tuple

from pybullet_utils.bullet_client import BulletClient

from racecar_gym.definitions import Position, Pose
from racecar_gym.models.configs import MapConfig


class Map:
    def __init__(self, client: BulletClient, config: MapConfig):
        self._client = client
        self._floor_id, self._walls_id, self._finish_id = None, None, None
        self._config = config
        self._starting_grid = [
            ((pose['x'], pose['y'], 0.25), (0.0, 0.0, pose['yaw']))
            for pose
            in config.starting_grid
        ]

        self._bounds = (config.lower_area_bounds, config.upper_area_bounds)

    @property
    def floor_id(self) -> int:
        assert self._floor_id is not None, 'reset() has to be called at least once'
        return self._floor_id

    @property
    def walls_id(self) -> int:
        assert self._walls_id is not None, 'reset() has to be called at least once'
        return self._walls_id

    @property
    def finish_id(self) -> int:
        assert self._finish_id is not None, 'reset() has to be called at least once'
        return self._finish_id

    @property
    def bounds(self) -> Tuple[Position, Position]:
        return self._bounds

    def starting_pose(self, position: int) -> Pose:
        assert position <= len(self._starting_grid), f'No position {position} available'
        position, orientation = self._starting_grid[position]
        return tuple(position), tuple(orientation)

    def reset(self):
        self._floor_id, self._walls_id, self._finish_id = self._client.loadSDF(self._config.sdf_file, globalScaling=1)
