from typing import List, Dict, Callable, Optional, Any, Tuple

from gymnasium.core import ObsType

from .single_agent_race import SingleAgentRaceEnv
from .multi_agent_race import MultiAgentRaceEnv
from racecar_gym.envs.scenarios import MultiAgentScenario, SingleAgentScenario
from .changing_track_race_env import ChangingTrackRaceEnv


class ChangingTrackMultiAgentRaceEnv(ChangingTrackRaceEnv):

    metadata = {'render_modes': ['human', 'rgb_array_follow', 'rgb_array_birds_eye']}

    def __init__(self, scenarios: List[str], render_mode: str = 'human', order: str = 'sequential', render_options: Dict = None):
        env_factories = [self._make_factory(scenario=s, render_mode=render_mode, render_options=render_options) for s in scenarios]
        super().__init__(env_factories, order)
        self._scenarios = scenarios

    @property
    def scenario(self):
        return self._scenarios[self._current_track_index]

    def _make_factory(self, scenario: str, render_mode: str, render_options: Dict) -> Callable[[], MultiAgentRaceEnv]:
        def factory():
            return MultiAgentRaceEnv(scenario=scenario, render_mode=render_mode, render_options=render_options)
        return factory

    def step(self, action: Dict):
        return super(ChangingTrackMultiAgentRaceEnv, self).step(action=action)

    def render(self):
        return super(ChangingTrackMultiAgentRaceEnv, self).render()

    def reset(self, *, seed: Optional[int] = None, options: Dict[str, Any] = None) -> Tuple[ObsType, Dict[str, Any]]:
        return super().reset(seed=seed, options=options)


class ChangingTrackSingleAgentRaceEnv(ChangingTrackRaceEnv):
    metadata = {'render_modes': ['human', 'rgb_array_follow', 'rgb_array_birds_eye']}

    def __init__(self, scenarios: List[str], render_mode: str = 'human', order: str = 'sequential',
                 render_options: Dict = None):
        env_factories = [self._make_factory(scenario=s, render_mode=render_mode, render_options=render_options) for s in
                         scenarios]
        super().__init__(env_factories, order)
        self._scenarios = scenarios

    @property
    def scenario(self):
        return self._scenarios[self._current_track_index]

    def _make_factory(self, scenario: str, render_mode: str, render_options: Dict) -> Callable[[], SingleAgentRaceEnv]:
        def factory():
            return SingleAgentRaceEnv(scenario=scenario, render_mode=render_mode, render_options=render_options)

        return factory

    def step(self, action: Dict):
        return super(ChangingTrackSingleAgentRaceEnv, self).step(action=action)

    def render(self, mode='follow', agent: str = None, **kwargs):
        return super(ChangingTrackSingleAgentRaceEnv, self).render()

    def reset(self, *, seed: Optional[int] = None, options: Dict[str, Any] = None) -> Tuple[ObsType, Dict[str, Any]]:
        return super().reset(seed=seed, options=options)





