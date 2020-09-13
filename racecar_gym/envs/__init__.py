import pybullet
from pybullet_utils.bullet_client import BulletClient

from .multi_race_car_env import MultiRaceCarEnv
from .specs import ScenarioSpec


def load_spec(path: str) -> ScenarioSpec:
    scenario = ScenarioSpec()
    scenario.load(path)
    return scenario


def load_from_spec(path: str) -> MultiRaceCarEnv:
    scenario = load_spec(path)
    if scenario.simulation_spec.rendering:
        client_factory = lambda: BulletClient(pybullet.GUI)
    else:
        client_factory = lambda: BulletClient(pybullet.DIRECT)

    return MultiRaceCarEnv(client_factory, scenario)
