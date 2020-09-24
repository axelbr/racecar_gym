from dataclasses import dataclass, field
from typing import List, Dict, Any

from yamldataclassconfig import YamlDataClassConfig


@dataclass
class SensorConfig(YamlDataClassConfig):
    type: str = None
    name: str = None
    params: Dict[str, Any] = None
    frequency: float = None


@dataclass
class ActuatorConfig(YamlDataClassConfig):
    type: str
    name: str
    params: Dict[str, Any] = None


@dataclass
class VehicleConfig(YamlDataClassConfig):
    urdf_file: str = None
    debug: bool = False
    actuators: List[ActuatorConfig] = field(default_factory=lambda: [])
    sensors: List[SensorConfig] = field(default_factory=lambda: [])


@dataclass
class MapConfig(YamlDataClassConfig):
    sdf_file: str = None
    wall_name: str = None
    segment_prefix: str = None
    starting_grid: List[Dict[str, float]] = None


@dataclass
class SimulationConfig(YamlDataClassConfig):
    time_step: float = None
    rendering: bool = None


@dataclass
class PhysicsConfig(YamlDataClassConfig):
    gravity: float = None


@dataclass
class SceneConfig(YamlDataClassConfig):
    name: str = None
    map: MapConfig = None
    physics: PhysicsConfig = None
    simulation: SimulationConfig = None
