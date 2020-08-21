import os
import yaml
from pybullet_utils.bullet_client import BulletClient

from racecar_gym.definitions import Pose
from racecar_gym.models.map import Map, MapConfig
from racecar_gym.models.racecar import RaceCar, RaceCarConfig


def load_map(client: BulletClient, config_file: str) -> Map:
    """
    Load map into simulation from configuration file.
    Args:
        client: A pybullet client instance.
        config_file: A map configuration file.

    Returns:
        A map which is loaded into the simulation.
    """
    with open(config_file) as f:
        config = yaml.safe_load(f)
    config['sdf_file'] = f"{os.path.dirname(config_file)}/{config['sdf_file']}"
    map_config = MapConfig(**config)
    return Map(client=client, config=map_config)

def load_vehicle(pose: Pose, client: BulletClient, config_file: str, debug: bool = False) -> RaceCar:
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
    with open(config_file) as f:
        config = yaml.safe_load(f)
    config['urdf_file'] = f"{os.path.dirname(config_file)}/{config['urdf_file']}"
    car_config = RaceCarConfig(
        debug=debug,
        urdf_file=config['urdf_file'],
        starting_pose=pose,
        motorized_joints=config['motorized_joints'],
        steering_joints=config['steering_joints'],
        lidar_joint=config['lidar_joint'],
        camera_joint=config['camera_joint'],

    )
    vehicle = RaceCar(client, config=car_config)
    return vehicle