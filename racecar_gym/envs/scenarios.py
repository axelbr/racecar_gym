from dataclasses import dataclass
from typing import Dict

from racecar_gym.bullet import load_world, load_vehicle
from racecar_gym.core import World
from racecar_gym.core.agent import Agent
from racecar_gym.core.specs import MultiAgentScenarioSpec
from racecar_gym.core.tasks import task_from_spec


@dataclass
class MultiAgentScenario:
    world: World
    agents: Dict[str, Agent]

    @staticmethod
    def from_spec(path: str, rendering: bool = None) -> 'MultiAgentScenario':
        spec = MultiAgentScenarioSpec()
        spec.load(path)
        if rendering:
            spec.world.rendering = rendering
        agents = dict([
            (s.id, Agent(id=s.id, vehicle=load_vehicle(s.vehicle), task=task_from_spec(s.task)))
            for s
            in spec.agents
        ])

        return MultiAgentScenario(world=load_world(spec.world), agents=agents)
