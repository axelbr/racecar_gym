from typing import List, Dict, Callable

from .single_agent_race import SingleAgentRaceEnv
from .multi_agent_race import MultiAgentRaceEnv
from .scenarios import MultiAgentScenario, SingleAgentScenario
from .util.changing_track_race_env import ChangingTrackRaceEnv
import pathlib

class ChangingTrackMultiAgentRaceEnv(ChangingTrackRaceEnv):

    def __init__(self, scenarios: List[MultiAgentScenario], order: str = 'sequential'):
        env_factories = [self._make_factory(scenario=s) for s in scenarios]
        super().__init__(env_factories, order)
        self._scenarios = scenarios

    @property
    def scenario(self):
        return self._scenarios[self._current_track_index]

    def _make_factory(self, scenario: MultiAgentScenario) -> Callable[[], MultiAgentRaceEnv]:
        def factory():
            return MultiAgentRaceEnv(scenario=scenario)
        return factory

    def step(self, action: Dict):
        return super(ChangingTrackMultiAgentRaceEnv, self).step(action=action)

    def render(self, mode='follow', agent: str = None, **kwargs):
        return super(ChangingTrackMultiAgentRaceEnv, self).render(mode=mode, agent=agent, **kwargs)


class ChangingTrackSingleAgentRaceEnv(ChangingTrackRaceEnv):

    def __init__(self, scenarios: List[SingleAgentScenario], order: str = 'sequential'):
        env_factories = [self._make_factory(s) for s in scenarios]
        super().__init__(env_factories, order)
        self._scenarios = scenarios

    @property
    def scenario(self):
        return self._scenarios[self._current_track_index]

    def step(self, action: Dict):
        return super(ChangingTrackSingleAgentRaceEnv, self).step(action=action)

    def _make_factory(self, scenario: SingleAgentScenario) -> Callable[[], SingleAgentRaceEnv]:
        def factory():
            return SingleAgentRaceEnv(scenario=scenario)
        return factory


