import math
from dataclasses import dataclass
from typing import Dict, Any

import gym
import pybullet as p

from racecar_gym.bullet.configs import MapConfig
from racecar_gym.core import world
from racecar_gym.core.definitions import Pose


class World(world.World):
    FLOOR_ID = 0
    WALLS_ID = 1
    FINISH_ID = 2

    @dataclass
    class Config:
        map_config: MapConfig
        rendering: bool
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
        if self._config.rendering:
            id = -1  # p.connect(p.SHARED_MEMORY)
            if id < 0:
                p.connect(p.GUI_SERVER)
        else:
            p.connect(p.DIRECT)

        floor_id, walls_id, finish_id = p.loadSDF(self._config.map_config.sdf_file)
        assert floor_id == World.FLOOR_ID and walls_id == World.WALLS_ID and finish_id == World.FINISH_ID, 'Objects not in correct order'
        p.setTimeStep(self._config.time_step)
        p.setGravity(0, 0, self._config.gravity)

    def reset(self):
        p.resetSimulation()
        floor_id, walls_id, finish_id = p.loadSDF(self._config.map_config.sdf_file)
        assert floor_id == World.FLOOR_ID and walls_id == World.WALLS_ID and finish_id == World.FINISH_ID, 'Objects not in correct order'
        p.setTimeStep(self._config.time_step)
        p.setGravity(0, 0, self._config.gravity)
        self._time = 0.0

    def initial_pose(self, position: int) -> Pose:
        assert position <= len(self._starting_grid), f'No position {position} available'
        position, orientation = self._starting_grid[position]
        return tuple(position), tuple(orientation)

    def update(self):
        p.stepSimulation()
        self._time += self._config.time_step

    def state(self) -> Dict[str, Any]:
        return {
            'time': self._time
        }

    def space(self) -> gym.Space:
        return gym.spaces.Dict({
            'time': gym.spaces.Box(low=0, high=math.inf, shape=(1,))
        })
