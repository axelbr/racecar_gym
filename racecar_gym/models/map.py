from dataclasses import dataclass
from typing import Tuple, List, Dict

from pybullet_utils.bullet_client import BulletClient

from racecar_gym.definitions import Position, Pose


@dataclass
class MapConfig:
    name: str
    sdf_file: str
    starting_grid: List[Dict[str, float]]
    area_bounds: Dict[str, Position]


class Map:
    def __init__(self, client: BulletClient, config: MapConfig):
        self._model = config.sdf_file
        self._client = client
        self._floor_id, self._walls_id, self._finish_id = self._load_map()
        self._config = config
        self._starting_grid = [
            ((pose['x'], pose['y'], 0.5), (0.0, 0.0, pose['yaw']))
            for pose
            in config.starting_grid
        ]

        self._bounds = (config.area_bounds['min'], config.area_bounds['max'])

    def _load_map(self) -> Tuple[int, int, int]:
        return self._client.loadSDF(self._model, globalScaling=1)

    @property
    def floor_id(self) -> int:
        return self._floor_id

    @property
    def walls_id(self) -> int:
        return self._walls_id

    @property
    def finish_id(self) -> int:
        return self._finish_id

    @property
    def bounds(self) -> Tuple[Position, Position]:
        return self._bounds

    def starting_pose(self, position: int) -> Pose:
        assert position <= len(self._starting_grid), f'No position {position} available'
        position, orientation = self._starting_grid[position]
        return tuple(position), tuple(orientation)

    def reset(self):
        self._floor_id, self._walls_id, self._finish_id = self._load_map()
