import os
import random
import urllib.request
import zipfile
from typing import List, Tuple

from racecar_gym import core
from racecar_gym.bullet.actuators import BulletActuator, Motor, SteeringWheel, Speed
from racecar_gym.bullet.configs import SensorConfig, VehicleConfig, ActuatorConfig, SceneConfig
from racecar_gym.bullet.sensors import Lidar, PoseSensor, AccelerationSensor, VelocitySensor, RGBCamera, BulletSensor, \
    FixedTimestepSensor
from racecar_gym.bullet.vehicle import RaceCar
from .world import World
from ..core.agent import Agent
from racecar_gym.core.specs import VehicleSpec, WorldSpec

base_path = os.path.dirname(os.path.abspath(__file__))


def load_sensor(config: SensorConfig) -> BulletSensor:
    if config.type == 'lidar':
        return Lidar(name=config.name, type=config.type, config=Lidar.Config(**config.params))
    if config.type == 'pose':
        return PoseSensor(name=config.name, type=config.type, config=PoseSensor.Config(**config.params))
    if config.type == 'acceleration':
        return AccelerationSensor(name=config.name, type=config.type, config=AccelerationSensor.Config(**config.params))
    if config.type == 'velocity':
        return VelocitySensor(name=config.name, type=config.type, config=VelocitySensor.Config(**config.params))
    if config.type == 'rgb_camera':
        return RGBCamera(name=config.name, type=config.type, config=RGBCamera.Config(**config.params))


def load_actuator(config: ActuatorConfig) -> BulletActuator:
    if config.type == 'motor':
        return Motor(name=config.name, config=Motor.Config(**config.params))
    if config.type == 'speed':
        return Speed(name=config.name, config=Speed.Config(**config.params))
    if config.type == 'steering':
        return SteeringWheel(name=config.name, config=SteeringWheel.Config(**config.params))


def _compute_color(name: str) -> Tuple[float, float, float, float]:
    return dict(
        red=(1.0, 0.0, 0.0, 1.0),
        green=(0.0, 1.0, 0.0, 1.0),
        blue=(0.0, 0.0, 1.0, 1.0),
        yellow=(1.0, 1.0, 0.0, 1.0),
        magenta=(1.0, 0.0, 1.0, 1.0)
    ).get(name, (random.random(), random.random(), random.random(), 1.0))


def load_vehicle(spec: VehicleSpec) -> core.Vehicle:
    config_file = f'{base_path}/../../models/vehicles/{spec.name}/{spec.name}.yml'
    if not os.path.exists(config_file):
        raise NotImplementedError(f'No vehicle with name {spec.name} implemented.')

    config = VehicleConfig()
    config.load(config_file)
    config.urdf_file = f'{os.path.dirname(config_file)}/{config.urdf_file}'
    config.color = spec.color
    requested_sensors = set(spec.sensors)
    available_sensors = set([sensor.name for sensor in config.sensors])

    if not requested_sensors.issubset(available_sensors):
        raise NotImplementedError(f'Sensors {requested_sensors - available_sensors} not available.')
    sensors = list(filter(lambda s: s.name in requested_sensors, config.sensors))
    sensors = [FixedTimestepSensor(sensor=load_sensor(config=c), frequency=c.frequency, time_step=0.01) for c in
               sensors]

    requested_actuators = set(spec.actuators)
    available_actuators = set([actuator.name for actuator in config.actuators])
    if not requested_actuators.issubset(available_actuators):
        raise NotImplementedError(f'Actuators {requested_actuators - available_actuators} not available.')
    actuators = list(filter(lambda a: a.name in requested_actuators, config.actuators))
    actuators = [load_actuator(config=c) for c in actuators]

    car_config = RaceCar.Config(urdf_file=config.urdf_file, color=_compute_color(config.color))
    vehicle = RaceCar(sensors=sensors, actuators=actuators, config=car_config)
    return vehicle


def load_world(spec: WorldSpec, agents: List[Agent]) -> core.World:
    scene_path = f'{base_path}/../../models/scenes'
    config_file = f'{scene_path}/{spec.name}/{spec.name}.yml'

    if not os.path.exists(config_file):
        try:
            print(f'Downloading {spec.name} track.')
            urllib.request.urlretrieve(
                f'https://github.com/axelbr/racecar_gym/releases/download/tracks-v1.0.0/{spec.name}.zip',
                f'{scene_path}/{spec.name}.zip'
            )
            with zipfile.ZipFile(f'{scene_path}/{spec.name}.zip', 'r') as zip:
                zip.extractall(f'{scene_path}/')
        except:
            raise NotImplementedError(f'No scene with name {spec.name} implemented.')

    config = SceneConfig()
    config.load(config_file)
    config.simulation.rendering = spec.rendering

    config.sdf = resolve_path(file=config_file, relative_path=config.sdf)
    config.map.maps = resolve_path(file=config_file, relative_path=config.map.maps)
    config.map.starting_grid = resolve_path(file=config_file, relative_path=config.map.starting_grid)

    world_config = World.Config(
        name=spec.name,
        sdf=config.sdf,
        map_config=config.map,
        time_step=config.simulation.time_step,
        gravity=config.physics.gravity,
        rendering=config.simulation.rendering,
    )

    return World(config=world_config, agents=agents)


def resolve_path(file: str, relative_path: str) -> str:
    file_dir = os.path.dirname(file)
    return f'{file_dir}/{relative_path}'
