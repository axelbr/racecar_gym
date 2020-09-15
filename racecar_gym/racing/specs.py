from dataclasses import dataclass, field
from typing import List, Dict, Any

from yamldataclassconfig import YamlDataClassConfig


@dataclass
class VehicleSpec(YamlDataClassConfig):
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
    implementation: str = None

@dataclass
class TaskSpec(YamlDataClassConfig):
    task_name: str = None
    params: Dict[str, Any] = field(default_factory=lambda: {})

@dataclass
class AgentSpec(YamlDataClassConfig):
    vehicle: VehicleSpec = VehicleSpec()
    task: TaskSpec = TaskSpec()


@dataclass
class ScenarioSpec(YamlDataClassConfig):
    laps: int = 2
    max_time: float = 120.0
    map: MapSpec = MapSpec()
    agents: List[AgentSpec] = field(default_factory=lambda: [])
    simulation: SimulationSpec = SimulationSpec()


@dataclass
class RuleSpec(YamlDataClassConfig):
    laps: int = None
    max_time: float = None


@dataclass
class MultiAgentScenarioSpec(YamlDataClassConfig):
    track: MapSpec = None
    agents: List[AgentSpec] = None
    rules: RuleSpec = None
    simulation: SimulationSpec = None
