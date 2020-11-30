import os
from gym.envs.registration import register
from .envs import VectorizedMultiAgentRaceEnv, MultiAgentScenario, SingleAgentScenario, SingleAgentRaceEnv, VectorizedSingleAgentRaceEnv
from .tasks import get_task, register_task, Task

base_path = os.path.dirname(__file__)

def _register(name: str, file: str, rendering: bool):
    scenario = MultiAgentScenario.from_spec(path=f'{base_path}/../scenarios/{file}', rendering=rendering)
    register(
        id=name,
        entry_point='racecar_gym.envs:MultiAgentRaceEnv',
        kwargs={'scenario': scenario}
    )

for scenario_file in os.listdir(f'{base_path}/../scenarios'):
    track_name = os.path.basename(scenario_file).split('.')[0]
    name = f'MultiAgent{track_name.capitalize()}'
    _register(name=f'{name}-v0', file=scenario_file, rendering=False)
    _register(name=f'{name}_Gui-v0', file=scenario_file, rendering=True)