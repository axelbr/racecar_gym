from dataclasses import dataclass

from pybullet_utils.bullet_client import BulletClient


@dataclass
class SimulationHandle:
    link_index: int
    body_id: int
    client: BulletClient