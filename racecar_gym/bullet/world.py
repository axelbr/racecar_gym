from dataclasses import dataclass

import pybullet as p

from racecar_gym.bullet.configs import MapConfig
from racecar_gym.entities import world
from racecar_gym.entities.definitions import Pose


class World(world.World):
    FLOOR_ID = 1
    WALLS_ID = 2
    FINISH_ID = 3

    @dataclass
    class Config:
        map_config: MapConfig
        time_step: float
        gravity: float

    def __init__(self, config: Config):
        self._config = config
        self._map_id = None
        self._time = 0.0
        self._starting_grid = [
            ((pose['x'], pose['y'], 0.25), (0.0, 0.0, pose['yaw']))
            for pose
            in config.map_config.starting_grid
        ]

    def init(self) -> None:
        floor_id, walls_id, finish_id = p.loadSDF(self._config.map_config.sdf_file)
        assert floor_id == World.FLOOR_ID and walls_id == World.WALLS_ID and finish_id == World.FINISH_ID, 'Objects not in correct order'
        p.setTimeStep(self._config.time_step)
        p.setGravity(0, 0, -self._config.gravity)

    def reset(self) -> None:
        p.resetSimulation()
        floor_id, walls_id, finish_id = p.loadSDF(self._config.map_config.sdf_file)
        assert floor_id == World.FLOOR_ID and walls_id == World.WALLS_ID and finish_id == World.FINISH_ID, 'Objects not in correct order'
        p.setTimeStep(self._config.time_step)
        p.setGravity(0, 0, -self._config.gravity)
        self._time = 0.0

    def initial_pose(self, position: int) -> Pose:
        assert position <= len(self._starting_grid), f'No position {position} available'
        position, orientation = self._starting_grid[position]
        return tuple(position), tuple(orientation)

    def update(self) -> float:
        p.stepSimulation()
        self._time += self._config.time_step
        return self._time
