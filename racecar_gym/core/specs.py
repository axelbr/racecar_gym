from dataclasses import dataclass, field
from typing import List, Dict, Any
from yamldataclassconfig.config import YamlDataClassConfig


@dataclass
class SimulationSpec(YamlDataClassConfig):
    time_step: float = 0.01
    rendering: bool = False
    implementation: str = None


@dataclass
class TaskSpec(YamlDataClassConfig):
    task_name: str = None
    params: Dict[str, Any] = field(default_factory=lambda: {})


@dataclass
class VehicleSpec(YamlDataClassConfig):
    name: str = None
    sensors: List[str] = field(default_factory=lambda: [])
    actuators: List[str] = field(default_factory=lambda: ['steering', 'motor'])
    color: str = 'blue' # either red, blue, green, magenta or random


@dataclass
class WorldSpec(YamlDataClassConfig):
    name: str = None
    rendering: bool = False


@dataclass
class AgentSpec(YamlDataClassConfig):
    id: str
    vehicle: VehicleSpec = VehicleSpec()
    task: TaskSpec = TaskSpec()


@dataclass
class ScenarioSpec(YamlDataClassConfig):
    world: WorldSpec = None
    agents: List[AgentSpec] = None
