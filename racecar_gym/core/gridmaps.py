from typing import Tuple

import numpy as np

from racecar_gym.core.definitions import Position


class GridMap:

    def __init__(self, grid_map: np.ndarray, resolution: float, origin: Position):
        self._resolution = resolution
        self._origin = origin
        self._map = grid_map
        self._height = grid_map.shape[0]
        self._width = grid_map.shape[1]

    def get_value(self, position: Position):
        origin_x, origin_y = self._origin[0], self._origin[1]
        x, y = position[0], position[1]
        px = int(self._height - (y - origin_y) / self._resolution)
        py = int((x - origin_x) / self._resolution)
        return self._map[px, py]

    def to_meter(self, px: int, py: int) -> Tuple[float, float]:
        origin_x, origin_y = self._origin[0], self._origin[1]
        y = origin_y - (px - self._height) * self._resolution
        x = py * self._resolution + origin_x
        return x, y

    @property
    def map(self):
        return self._map

