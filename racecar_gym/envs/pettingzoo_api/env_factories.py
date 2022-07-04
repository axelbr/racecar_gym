from pettingzoo import AECEnv, ParallelEnv

from ..scenarios import MultiAgentScenario
from .racecarenv import _MultiAgentEnv

def env(scenario_path: str, reset_mode: str = 'grid', live_rendering: bool = False) -> AECEnv:
    return raw_env(scenario_path=scenario_path, reset_mode=reset_mode, live_rendering=live_rendering)

def raw_env(scenario_path: str, reset_mode: str = 'grid', live_rendering: bool = False) -> AECEnv:
    scenario = MultiAgentScenario.from_spec(path=scenario_path, rendering=live_rendering)
    return _MultiAgentEnv(scenario=scenario, reset_mode=reset_mode)