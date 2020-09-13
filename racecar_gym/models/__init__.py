import os

from pybullet_utils.bullet_client import BulletClient
import racecar_gym
from racecar_gym.models.configs import VehicleConfig, MapConfig
from racecar_gym.models.map import Map
from racecar_gym.models.racecar import RaceCar

base_path = os.path.dirname(racecar_gym.__file__)

def load_map(client: BulletClient, config_file: str) -> Map:
    """
    Load map into simulation from configuration file.
    Args:
        client: A pybullet client instance.
        config_file: A map configuration file.

    Returns:
        A map which is loaded into the simulation.
    """
    map_config = MapConfig()
    map_config.load(f'{base_path}/../{config_file}')
    map_config.sdf_file = f'{base_path}/../{os.path.dirname(config_file)}/{map_config.sdf_file}'
    print(map_config.sdf_file)
    return Map(client=client, config=map_config)


def load_vehicle(client: BulletClient, map: Map, config_file: str) -> RaceCar:
    """
    Loads a vehicle into simulation from a config file.
    Args:
        pose: Initial pose of a vehicle.
        client: A pybullet client instance.
        config_file: A vehicle configuration file.
        debug: Debug flag for additional visualizations (lidar rays)

    Returns:
        Vehicle which is loaded into the simulation.
    """
    car_config = VehicleConfig()
    car_config.load(f'{base_path}/../{config_file}')
    car_config.urdf_file = f'{base_path}/../{os.path.dirname(config_file)}/{car_config.urdf_file}'
    vehicle = RaceCar(client=client, map=map, config=car_config)
    return vehicle
