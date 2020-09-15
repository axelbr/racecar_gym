from dataclasses import dataclass, field
from typing import List, Dict, Any

from yamldataclassconfig import YamlDataClassConfig

from racecar_gym.entities.definitions import Position


@dataclass
class SensorConfig(YamlDataClassConfig):
    type: str = None
    name: str = None
    link: str = None
    params: Dict[str, Any] = None
    frequency: int = None


@dataclass
class VehicleJointConfig(YamlDataClassConfig):
    motorized_joints: List[str] = field(default_factory=lambda: [])
    steering_joints: List[str] = field(default_factory=lambda: [])
    lidar_joint: List[str] = field(default_factory=lambda: [])
    camera_joint: List[str] = field(default_factory=lambda: [])


@dataclass
class VehicleConfig(YamlDataClassConfig):
    urdf_file: str = None
    max_speed: float = 14.0
    max_steering_angle: float = 0.42
    max_force: float = 5.0
    speed_multiplier: float = 20.0
    steering_multiplier: float = 0.5
    debug: bool = False
    sensors: List[SensorConfig] = field(default_factory=lambda: [])
    joints: VehicleJointConfig = VehicleJointConfig()


@dataclass
class MapConfig(YamlDataClassConfig):
    name: str = None
    sdf_file: str = None
    starting_grid: List[Dict[str, float]] = None
    lower_area_bounds: Position = None
    upper_area_bounds: Position = None

