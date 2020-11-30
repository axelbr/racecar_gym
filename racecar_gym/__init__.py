import os
from gym.envs.registration import register
from .envs import VectorizedMultiAgentRaceEnv, MultiAgentScenario, SingleAgentScenario, SingleAgentRaceEnv, VectorizedSingleAgentRaceEnv
from .tasks import get_task, register_task, Task

base_path = os.path.dirname(__file__)

def _register(name: str, file: str, is_multi_agent: bool, rendering: bool):
    if is_multi_agent:
        scenario = MultiAgentScenario.from_spec(path=f'{base_path}/../scenarios/{file}', rendering=rendering)
        entry_point = 'racecar_gym.envs:MultiAgentRaceEnv'
    else:
        scenario = SingleAgentScenario.from_spec(path=f'{base_path}/../scenarios/{file}', rendering=rendering)
        entry_point = 'racecar_gym.envs:SingleAgentRaceEnv'
    register(
        id=name,
        entry_point=entry_point,
        kwargs={'scenario': scenario}
    )

for scenario_file in os.listdir(f'{base_path}/../scenarios'):
    type_track = os.path.basename(scenario_file).split('.')[0].split("_", 1)[0]    # `single` or `multi`
    track_name = os.path.basename(scenario_file).split('.')[0].split("_", 1)[1]    # `austria`, `berlin`, ...
    if type_track == "single":
        name = f'SingleAgent{track_name.capitalize()}'
        is_multi_agent = False
    elif type_track == "multi":
        name = f'MultiAgent{track_name.capitalize()}'
        is_multi_agent = True
    else:
        raise NotImplementedError(f"type track {type_track} not supported")
    _register(name=f'{name}-v0', file=scenario_file, is_multi_agent=is_multi_agent, rendering=False)
    _register(name=f'{name}_Gui-v0', file=scenario_file, is_multi_agent=is_multi_agent, rendering=True)