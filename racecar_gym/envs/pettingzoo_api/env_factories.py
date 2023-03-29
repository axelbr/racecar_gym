from typing import Optional, Dict

from pettingzoo import ParallelEnv

from .racecarenv import _MultiAgentRaceEnv


def env(scenario: str, render_mode: str ='human', render_options: Optional[Dict] = None) -> ParallelEnv:
    return raw_env(scenario, render_mode, render_options)

def raw_env(scenario: str, render_mode: str ='human', render_options: Optional[Dict] = None) -> ParallelEnv:
    return _MultiAgentRaceEnv(scenario=scenario, render_mode=render_mode, render_options=render_options)