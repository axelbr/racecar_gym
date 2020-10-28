import math
import random
from dataclasses import dataclass
from typing import Dict, Any, List

import gym
import numpy as np
import pybullet as p

from racecar_gym.bullet.configs import MapConfig
from racecar_gym.core import world
from racecar_gym.core.agent import Agent
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
        start_positions: str

    def __init__(self, config: Config, agents: List[Agent]):
        self._config = config
        self._map_id = None
        self._time = 0.0
        self._agents = agents
        self._collisions = dict([(a.id, False) for a in agents])
        self._progress = dict([(a.id, (0, 0)) for a in agents])
        self._laps = dict([(a.id, 0) for a in agents])
        self._objects = {}
        self._starting_grid = [
            ((pose['x'], pose['y'], 0.25), (0.0, 0.0, pose['yaw']))
            for pose
            in config.map_config.starting_grid
        ]

    def init(self) -> None:
        if self._config.rendering:
            id = -1  # p.connect(p.SHARED_MEMORY)
            if id < 0:
                p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        self._load_scene(self._config.map_config.sdf_file)
        p.setTimeStep(self._config.time_step)
        p.setGravity(0, 0, self._config.gravity)

    def reset(self):
        #p.resetSimulation()
        #self._load_scene(self._config.map_config.sdf_file)
        p.setTimeStep(self._config.time_step)
        p.setGravity(0, 0, self._config.gravity)
        p.stepSimulation()
        self._time = 0.0
        self._collisions = dict([(a.id, False) for a in self._agents])
        self._progress = dict([(a.id, (None, 0)) for a in self._agents])
        self._laps = dict([(a.id, 0) for a in self._agents])

    def _load_scene(self, sdf_file: str):
        ids = p.loadSDF(sdf_file)
        objects = dict([(p.getBodyInfo(i)[1].decode('ascii'), i) for i in ids])
        self._objects['wall'] = objects[self._config.map_config.wall_name]
        segment_ids = filter(
            lambda name: name.startswith(self._config.map_config.segment_prefix),
            objects.keys()
        )

        self._objects['segments'] = dict([(objects[id], i) for i, id in enumerate(segment_ids)])

    def get_starting_position(self, agent: Agent) -> Pose:
        if self._config.start_positions == 'index':
            position = list(map(lambda agent: agent.id, self._agents)).index(agent.id)
            position, orientation = self._starting_grid[position]
            return tuple(position), tuple(orientation)
        if self._config.start_positions == 'random':
            segments = list(self._objects['segments'].values())
            section = random.choice(segments)
            next_section = section + 1 if section < max(segments) else min(segments)
            section = p.getAABB(section)
            next_section = p.getAABB(next_section)
            position = (np.array(section[1]) + np.array(section[0])) / 2
            next_position = (np.array(next_section[1]) + np.array(next_section[0])) / 2
            diff = next_position - position
            angle = np.arctan2(diff[1], diff[0])
            angle = np.random.normal(loc=angle, scale=0.15)
            position[2] = 0.1
            return tuple(position), (0, 0, angle)
        raise NotImplementedError(self._config.start_positions)

    def update(self):
        p.stepSimulation()
        self._time += self._config.time_step

    def state(self) -> Dict[str, Any]:

        for agent in self._agents:
            self._update_race_info(agent=agent)

        state = {}

        for agent in self._agents:
            agent_state = {
                'collision': self._collisions[agent.id],
                'section': self._progress[agent.id][0],
                'section_time': self._progress[agent.id][1],
                'lap': self._laps[agent.id],
                'time': self._time,
                'n_segments': len(self._objects["segments"])
            }

            state[agent.id] = agent_state

        return state

    def space(self) -> gym.Space:
        return gym.spaces.Dict({
            'time': gym.spaces.Box(low=0, high=math.inf, shape=(1,))
        })

    def _update_race_info(self, agent):
        contact_points = set([c[2] for c in p.getContactPoints(agent.vehicle_id)])
        segment = None
        collision = False
        for contact in contact_points:
            if self._objects['wall'] == contact:
                collision = True
            elif contact in self._objects['segments']:
                segment = min(segment, contact) if segment else contact
            else:
                collision = True

        self._collisions[agent.id] = collision

        if len(contact_points) > 0:
            current_progress = self._progress[agent.id][0]
            if current_progress is not None and segment is not None:
                if segment == 1 and current_progress == len(self._objects['segments']):
                    self._laps[agent.id] += 1
                    self._progress[agent.id] = (1, self._time)
                elif segment > current_progress:
                    self._progress[agent.id] = (current_progress + 1, self._time)
            else:
                self._progress[agent.id] = (segment, self._time)