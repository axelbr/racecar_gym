import os
from gym.envs.registration import register
from .envs import VectorizedMultiAgentRaceEnv, MultiAgentScenario, SingleAgentScenario, SingleAgentRaceEnv, VectorizedSingleAgentRaceEnv
from .tasks import get_task, register_task, Task

base_path = os.path.dirname(__file__)

def _register_multi_agent(name: str, file: str, rendering: bool):
    scenario = MultiAgentScenario.from_spec(path=f'{base_path}/../scenarios/{file}', rendering=rendering)
    register(
        id=name,
        entry_point='racecar_gym.envs:MultiAgentRaceEnv',
        kwargs={'scenario': scenario}
    )

def _register_single_agent(name: str, file: str, rendering: bool):
    scenario = SingleAgentScenario.from_spec(path=f'{base_path}/../scenarios/{file}', rendering=rendering)
    register(
        id=name,
        entry_point='racecar_gym.envs:SingleAgentRaceEnv',
        kwargs={'scenario': scenario}
    )

for scenario_file in os.listdir(f'{base_path}/../scenarios'):
    track_name = os.path.basename(scenario_file).split('.')[0]
    name = f'{track_name.capitalize()}'
    _register_multi_agent(name=f'MultiAgent{name}-v0', file=scenario_file, rendering=False)
    _register_multi_agent(name=f'MultiAgent{name}_Gui-v0', file=scenario_file, rendering=True)
    _register_single_agent(name=f'SingleAgent{name}-v0', file=scenario_file, rendering=False)
    _register_single_agent(name=f'SingleAgent{name}_Gui-v0', file=scenario_file, rendering=True)


