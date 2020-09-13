from dataclasses import dataclass, field
from typing import Union, List, Dict, Any

from yamldataclassconfig import YamlDataClassConfig


@dataclass
class MultiRaceCarScenario:
    map: str
    cars: Union[List[str], str]
    time_step: float = 0.01
    no_of_cars: int = 1
    max_speed: float = 14.0
    max_steering_angle: float = 0.4
    lidar_rays: int = 100
    lidar_range: float = 5
    start_gui: bool = False
    laps: int = 2
    max_time: float = 120.0


@dataclass
class VehicleSpec(YamlDataClassConfig):
    config_file: str = None
    name: str = None
    sensors: List[str] = field(default_factory=lambda: [])


@dataclass
class MapSpec(YamlDataClassConfig):
    config_file: str = None
    name: str = None

@dataclass
class SimulationSpec(YamlDataClassConfig):
    time_step: float = 0.01
    rendering: bool = False

@dataclass
class TaskSpec(YamlDataClassConfig):
    task_name: str = None
    params: Dict[str, Any] = field(default_factory=lambda: {})

@dataclass
class ScenarioSpec(YamlDataClassConfig):
    laps: int = 2
    max_time: float = 120.0
    map_spec: MapSpec = MapSpec()
    task_spec: TaskSpec = TaskSpec()
    vehicle_spec: List[VehicleSpec] = field(default_factory=lambda: [])
    simulation_spec: SimulationSpec = SimulationSpec()
