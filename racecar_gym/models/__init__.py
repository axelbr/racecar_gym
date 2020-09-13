import os

from pybullet_utils.bullet_client import BulletClient
import racecar_gym
from racecar_gym.models.configs import VehicleConfig, MapConfig
from racecar_gym.models.map import Map
from racecar_gym.models.racecar import RaceCar

base_path = os.path.dirname(racecar_gym.__file__)

def load_map_from_config(client: BulletClient, config_file: str) -> Map:
    map_config = MapConfig()
    map_config.load(config_file)
    map_config.sdf_file = f'{os.path.dirname(config_file)}/{map_config.sdf_file}'
    return Map(client=client, config=map_config)

def load_map_by_name(client: BulletClient, map_name: str) -> Map:
    """
    Load map into simulation from configuration file.
    Args:
        client: A pybullet client instance.
        config_file: A map configuration file.

    Returns:
        A map which is loaded into the simulation.
    """
    path = f'{base_path}/../models/tracks/{map_name}'
    if os.path.exists(path):
        map_config = MapConfig()
        map_config.load(f'{path}/{map_name}.yml')
        map_config.sdf_file = f'{path}/{map_config.sdf_file}'
        print(map_config.sdf_file)
        return Map(client=client, config=map_config)
    else:
        raise FileNotFoundError(f'No such map: {map}')


def load_vehicle_by_name(client: BulletClient, map: Map, vehicle: str) -> RaceCar:
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
    path = f'{base_path}/../models/cars/{vehicle}'
    if os.path.exists(path):
        car_config = VehicleConfig()
        car_config.load(f'{path}/{vehicle}.yml')
        car_config.urdf_file = f'{path}/{car_config.urdf_file}'
        vehicle = RaceCar(client=client, map=map, config=car_config)
        return vehicle
    else:
        raise FileNotFoundError(f'No such vehicle: {vehicle}')

def load_vehicle_from_config(client: BulletClient, map: Map, config_file: str) -> RaceCar:
    car_config = VehicleConfig()
    car_config.load(config_file)
    car_config.urdf_file = f'{os.path.dirname(config_file)}/{car_config.urdf_file}'
    vehicle = RaceCar(client=client, map=map, config=car_config)
    return vehicle