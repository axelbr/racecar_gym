from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple

from yamldataclassconfig.config import YamlDataClassConfig


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
    color: str = 'blue'
    debug: bool = False
    actuators: List[ActuatorConfig] = field(default_factory=lambda: [])
    sensors: List[SensorConfig] = field(default_factory=lambda: [])


@dataclass
class MapConfig(YamlDataClassConfig):
    resolution: float = None
    origin: List[float] = None
    maps: str = None
    starting_grid: str = None
    checkpoints: int = None


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
    sdf: str = None
    map: MapConfig = None
    physics: PhysicsConfig = None
    simulation: SimulationConfig = None
