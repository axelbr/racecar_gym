import os
from typing import List

from racecar_gym import core
from racecar_gym.bullet.actuators import BulletActuator, Motor, SteeringWheel
from racecar_gym.bullet.configs import SensorConfig, VehicleConfig, ActuatorConfig, SceneConfig
from racecar_gym.bullet.sensors import Lidar, GPS, IMU, Tachometer, RGBCamera, BulletSensor, FixedTimestepSensor
from racecar_gym.bullet.vehicle import RaceCar
from .world import World
from ..core.agent import Agent
from racecar_gym.envs.specs import WorldSpec, VehicleSpec

base_path = os.path.dirname(os.path.abspath(__file__))

def load_sensor(config: SensorConfig) -> BulletSensor:
    if config.type == 'lidar':
        return Lidar(name=config.name, type=config.type, config=Lidar.Config(**config.params))
    if config.type == 'gps':
        return GPS(name=config.name, type=config.type, config=GPS.Config(**config.params))
    if config.type == 'imu':
        return IMU(name=config.name, type=config.type, config=IMU.Config(**config.params))
    if config.type == 'tacho':
        return Tachometer(name=config.name, type=config.type, config=Tachometer.Config(**config.params))
    if config.type == 'rgb_camera':
        return RGBCamera(name=config.name, type=config.type, config=RGBCamera.Config(**config.params))


def load_actuator(config: ActuatorConfig) -> BulletActuator:
    if config.type == 'motor':
        return Motor(name=config.name, config=Motor.Config(**config.params))
    if config.type == 'steering':
        return SteeringWheel(name=config.name, config=SteeringWheel.Config(**config.params))


def load_vehicle(spec: VehicleSpec) -> core.Vehicle:
    config_file = f'{base_path}/../../models/vehicles/{spec.name}/{spec.name}.yml'
    if not os.path.exists(config_file):
        raise NotImplementedError(f'No vehicle with name {spec.name} implemented.')

    config = VehicleConfig()
    config.load(config_file)
    config.urdf_file = f'{os.path.dirname(config_file)}/{config.urdf_file}'

    requested_sensors = set(spec.sensors)
    available_sensors = set([sensor.name for sensor in config.sensors])

    if not requested_sensors.issubset(available_sensors):
        raise NotImplementedError(f'Sensors {requested_sensors - available_sensors} not available.')
    sensors = list(filter(lambda s: s.name in requested_sensors, config.sensors))
    sensors = [FixedTimestepSensor(sensor=load_sensor(config=c), frequency=c.frequency, time_step=0.01) for c in
               sensors]
    actuators = [load_actuator(config=c) for c in config.actuators]
    car_config = RaceCar.Config(urdf_file=config.urdf_file)
    vehicle = RaceCar(sensors=sensors, actuators=actuators, config=car_config)
    return vehicle


def load_world(spec: WorldSpec, agents: List[Agent]) -> core.World:
    config_file = f'{base_path}/../../models/scenes/{spec.name}/{spec.name}.yml'
    if not os.path.exists(config_file):
        raise NotImplementedError(f'No scene with name {spec.name} implemented.')

    config = SceneConfig()
    config.load(config_file)
    config.simulation.rendering = spec.rendering

    config.sdf = resolve_path(file=config_file, relative_path=config.sdf)
    config.map.maps = resolve_path(file=config_file, relative_path=config.map.maps)
    config.map.starting_grid = resolve_path(file=config_file, relative_path=config.map.starting_grid)


    world_config = World.Config(
        sdf=config.sdf,
        map_config=config.map,
        time_step=config.simulation.time_step,
        gravity=config.physics.gravity,
        rendering=config.simulation.rendering,
        start_positions=spec.start_positions
    )

    return World(config=world_config, agents=agents)


def resolve_path(file: str, relative_path: str) -> str:
    file_dir = os.path.dirname(file)
    return f'{file_dir}/{relative_path}'