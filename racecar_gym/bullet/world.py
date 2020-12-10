import math
import random
from dataclasses import dataclass
from typing import Dict, Any, List

import gym
import numpy as np
import pybullet as p

from racecar_gym.bullet import util
from racecar_gym.bullet.configs import MapConfig
from racecar_gym.core import world
from racecar_gym.core.agent import Agent
from racecar_gym.core.definitions import Pose
from racecar_gym.core.gridmaps import GridMap


class World(world.World):
    FLOOR_ID = 0
    WALLS_ID = 1
    FINISH_ID = 2

    @dataclass
    class Config:
        sdf: str
        map_config: MapConfig
        rendering: bool
        time_step: float
        gravity: float

    def __init__(self, config: Config, agents: List[Agent]):
        self._config = config
        self._map_id = None
        self._time = 0.0
        self._agents = agents
        self._state = dict([(a.id, {}) for a in agents])
        self._objects = {}
        self._starting_grid = np.load(config.map_config.starting_grid)['data']
        self._maps = dict([
            (name, GridMap(
                grid_map=np.load(config.map_config.maps)[data],
                origin=self._config.map_config.origin,
                resolution=self._config.map_config.resolution
            ))
            for name, data
            in [
                ('progress', 'norm_distance_from_start'),
                ('obstacle', 'norm_distance_to_obstacle'),
                ('occupancy', 'drivable_area')
                ]
        ])

        self._state['maps'] = self._maps


    def init(self) -> None:
        if self._config.rendering:
            id = -1  # p.connect(p.SHARED_MEMORY)
            if id < 0:
                p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        self._load_scene(self._config.sdf)
        p.setTimeStep(self._config.time_step)
        p.setGravity(0, 0, self._config.gravity)

    def reset(self):
        p.setTimeStep(self._config.time_step)
        p.setGravity(0, 0, self._config.gravity)
        p.stepSimulation()
        self._time = 0.0
        self._state = dict([(a.id, {}) for a in self._agents])

    def _load_scene(self, sdf_file: str):
        ids = p.loadSDF(sdf_file)
        objects = dict([(p.getBodyInfo(i)[1].decode('ascii'), i) for i in ids])
        self._objects = objects

    def get_starting_position(self, agent: Agent, mode: str) -> Pose:
        if mode == 'grid':
            position = list(map(lambda agent: agent.id, self._agents)).index(agent.id)
            pose = self._starting_grid[position]
            return tuple(pose[:3]), tuple(pose[3:])
        if mode == 'checkpoint':
            pass
        if mode == 'random':
            distance_map = self._maps['obstacle']
            progress_map = self._maps['progress']
            center_corridor = np.argwhere(distance_map.map > 0.5)
            x, y, angle = self._random_position(progress_map, center_corridor)
            return (x, y, 0.05), (0, 0, angle)
        if mode == 'biased_random':
            distance_map = self._maps['obstacle']
            progress_map = self._maps['progress']
            if random.random() < 0.5:
                sampling_area = np.argwhere((distance_map.map > 0.5) &
                                            (0.25 <= progress_map.map) & (progress_map.map <= 0.30))
            else:
                sampling_area = np.argwhere(distance_map.map > 0.5)
            x, y, angle = self._random_position(progress_map, sampling_area)
            return (x, y, 0.05), (0, 0, angle)
        raise NotImplementedError(mode)

    def _random_position(self, progress_map, sampling_map, delta_progress_next_pos=0.025):
        position = random.choice(sampling_map)
        progress = progress_map.map[position[0], position[1]]

        # next position is a random point in the map which has progress greater than the current `position`
        # note: this approach suffers in "wide" tracks
        direction_progress_min = (progress + delta_progress_next_pos) % 1
        direction_progress_max = (progress + 2 * delta_progress_next_pos) % 1
        if direction_progress_min < direction_progress_max:
            direction_area = np.argwhere(np.logical_and(
                progress_map.map > direction_progress_min,
                progress_map.map <= direction_progress_max,
            ))
        else:
            # bugfix: when min=0.99, max=0.01, the 'and' is empty
            direction_area = np.argwhere(np.logical_or(
                progress_map.map <= direction_progress_min,
                progress_map.map > direction_progress_max,
            ))
        if direction_area.shape[0] == 0:
            raise ValueError(f"starting position not exist, consider to change `delta_progress`={delta_progress_next_pos}")
        next_position = random.choice(direction_area)
        px, py = position[0], position[1]
        npx, npy = next_position[0], next_position[1]
        diff = np.array(progress_map.to_meter(npx, npy)) - np.array(progress_map.to_meter(px, py))
        angle = np.arctan2(diff[1], diff[0])
        x, y = progress_map.to_meter(px, py)
        return x, y, angle




    def update(self):
        p.stepSimulation()
        self._time += self._config.time_step

    def state(self) -> Dict[str, Any]:
        for agent in self._agents:
            self._update_race_info(agent=agent)

        self._update_ranks()

        return self._state

    def space(self) -> gym.Space:
        return gym.spaces.Dict({
            'time': gym.spaces.Box(low=0, high=math.inf, shape=(1,))
        })

    def _update_race_info(self, agent):
        contact_points = set([c[2] for c in p.getContactPoints(agent.vehicle_id)])
        progress_map = self._maps['progress']
        obstacle_map = self._maps['obstacle']
        self._state[agent.id]['pose'] = util.get_pose(id=agent.vehicle_id)

        collision_with_wall = False
        opponent_collisions = []
        opponents = dict([(a.vehicle_id, a.id) for a in self._agents])
        for contact in contact_points:
            if self._objects['walls'] == contact:
                collision_with_wall = True
            elif contact in opponents:
                opponent_collisions.append(opponents[contact])

        self._state[agent.id]['wall_collision'] = collision_with_wall
        self._state[agent.id]['opponent_collisions'] = opponent_collisions
        velocity = util.get_velocity(id=agent.vehicle_id)

        if 'velocity' in self._state[agent.id]:
            previous_velocity = self._state[agent.id]['velocity']
            self._state[agent.id]['acceleration'] = (velocity - previous_velocity) / self._config.time_step
        else:
            self._state[agent.id]['acceleration'] = velocity / self._config.time_step

        pose = self._state[agent.id]['pose']
        progress = progress_map.get_value(position=(pose[0], pose[1], 0))
        dist_obstacle = obstacle_map.get_value(position=(pose[0], pose[1], 0))
        self._state[agent.id]['velocity'] = velocity
        self._state[agent.id]['progress'] = progress
        self._state[agent.id]['obstacle'] = dist_obstacle
        self._state[agent.id]['time'] = self._time

        progress = self._state[agent.id]['progress']
        checkpoints = 1.0 / float(self._config.map_config.checkpoints)

        checkpoint = int(progress / checkpoints)

        if 'checkpoint' in self._state[agent.id]:
            last_checkpoint = self._state[agent.id]['checkpoint']
            if last_checkpoint + 1 == checkpoint:
                self._state[agent.id]['checkpoint'] = checkpoint
                self._state[agent.id]['wrong_way'] = False
            elif last_checkpoint - 1 == checkpoint:
                self._state[agent.id]['wrong_way'] = True
            elif last_checkpoint == self._config.map_config.checkpoints and checkpoint == 0:
                self._state[agent.id]['lap'] += 1
                self._state[agent.id]['checkpoint'] = checkpoint
                self._state[agent.id]['wrong_way'] = False
            elif last_checkpoint == 0 and checkpoint == self._config.map_config.checkpoints:
                self._state[agent.id]['wrong_way'] = True
        else:
            self._state[agent.id]['checkpoint'] = checkpoint
            self._state[agent.id]['lap'] = 1
            self._state[agent.id]['wrong_way'] = False

    def _update_ranks(self):

        agents = [
            (agent_id, self._state[agent_id]['lap'], self._state[agent_id]['progress'])
            for agent_id
            in map(lambda a: a.id, self._agents)
        ]

        ranked = [item[0] for item in sorted(agents, key=lambda item: (item[1], item[2]), reverse=True)]

        for agent in self._agents:
            rank = ranked.index(agent.id) + 1
            self._state[agent.id]['rank'] = rank

    def render(self, agent_id: str, mode: str) -> np.ndarray:
        agent = list(filter(lambda a: a.id == agent_id, self._agents))
        assert len(agent) == 1
        agent = agent[0]
        if mode == 'follow':
            return util.follow_agent(agent=agent)
        elif mode == 'birds_eye':
            return util.birds_eye(agent=agent)
