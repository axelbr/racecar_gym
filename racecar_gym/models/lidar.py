import math
from time import time
from typing import Dict, Tuple, List, Any

from gym import spaces
import numpy as np
from pybullet_utils.bullet_client import BulletClient


class Lidar:

    def __init__(self, rays: int, min_range: float, max_range: float, client: BulletClient, car_id: int, id: int, rendering: bool = False):
        self._rays = rays
        self._client = client
        self._id = id
        self._car_id = car_id
        self._render = rendering
        self._min_range = min_range
        self._max_range = max_range
        self._range = max_range - min_range
        self._hit_color = [1, 0, 0]
        self._miss_color = [0, 1, 0]
        self._ray_from = []
        self._ray_to = []
        self._ray_ids = []
        self._last_scan_time = time()
        self._setup_rays()

    @property
    def last_scan_time(self):
        return self._last_scan_time

    def _setup_rays(self):
        for i in range(self._rays):

            self._ray_from.append([
                self._min_range * math.sin(-0.5 * 0.25 * 2. * math.pi + 0.75 * 2. * math.pi * float(i) / self._rays),
                self._min_range * math.cos(-0.5 * 0.25 * 2. * math.pi + 0.75 * 2. * math.pi * float(i) / self._rays),
                0
            ])

            self._ray_to.append([
                self._max_range * math.sin(-0.5 * 0.25 * 2. * math.pi + 0.75 * 2. * math.pi * float(i) / self._rays),
                self._max_range * math.cos(-0.5 * 0.25 * 2. * math.pi + 0.75 * 2. * math.pi * float(i) / self._rays),
                0
            ])

            if self._render:
                ray_id = self._client.addUserDebugLine(self._ray_from[i], self._ray_to[i], self._miss_color,
                                                       parentObjectUniqueId=self._car_id,
                                                       parentLinkIndex=self._id)
                self._ray_ids.append(ray_id)

        results = self._client.rayTestBatch(self._ray_from, self._ray_to, 0, parentObjectUniqueId=self._car_id, parentLinkIndex=self._id)

        if self._render:
            for i in range(self._rays):
                hitFraction = results[i][2]
                if (hitFraction == 1.):
                    self._client.addUserDebugLine(self._ray_from[i], self._ray_to[i], self._miss_color, replaceItemUniqueId=self._ray_ids[i],
                                                  parentObjectUniqueId=self._car_id, parentLinkIndex=self._id)
                else:
                    localHitTo = [self._ray_from[i][0] + hitFraction * (self._ray_to[i][0] - self._ray_from[i][0]),
                                  self._ray_from[i][1] + hitFraction * (self._ray_to[i][1] - self._ray_from[i][1]),
                                  self._ray_from[i][2] + hitFraction * (self._ray_to[i][2] - self._ray_from[i][2])]
                    self._client.addUserDebugLine(self._ray_from[i], localHitTo, self._hit_color, replaceItemUniqueId=self._ray_ids[i],
                                                  parentObjectUniqueId=self._car_id, parentLinkIndex=self._id)
        self._last_scan_time = time()

    def _visualize(self, ray: int, hit_fraction: float):
        if (hit_fraction == 1.):
            self._client.addUserDebugLine(self._ray_from[ray], self._ray_to[ray], self._miss_color,
                                          replaceItemUniqueId=self._ray_ids[ray], parentObjectUniqueId=self._car_id,
                                          parentLinkIndex=self._id)
        else:
            localHitTo = [self._ray_from[ray][0] + hit_fraction * (self._ray_to[ray][0] - self._ray_from[ray][0]),
                          self._ray_from[ray][1] + hit_fraction * (self._ray_to[ray][1] - self._ray_from[ray][1]),
                          self._ray_from[ray][2] + hit_fraction * (self._ray_to[ray][2] - self._ray_from[ray][2])]

            self._client.addUserDebugLine(self._ray_from[ray], localHitTo, self._hit_color,
                                          replaceItemUniqueId=self._ray_ids[ray],
                                          parentObjectUniqueId=self._car_id, parentLinkIndex=self._id)

    def scan(self) -> np.ndarray:
        results = self._client.rayTestBatch(self._ray_from, self._ray_to, 0, parentObjectUniqueId=self._car_id, parentLinkIndex=self._id)
        scan = np.full(self._rays, self._max_range)
        for i in range(self._rays):
            hit_fraction = results[i][2]
            scan[i] = self._range * hit_fraction
            if self._render:
                self._visualize(ray=i, hit_fraction=hit_fraction)
        self._last_scan_time = time()
        return scan

    def reset(self):
        self._setup_rays()