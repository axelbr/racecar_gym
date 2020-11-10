import os
from gym.envs.registration import register
from .envs import VectorizedMultiAgentRaceEnv, MultiAgentScenario, SingleAgentScenario, SingleAgentRaceEnv, VectorizedSingleAgentRaceEnv
from .tasks import get_task, register_task, Task

base_path = os.path.dirname(__file__)

register(id='MultiAgentAustria_Gui-v0',
    entry_point='racecar_gym.envs:MultiAgentRaceEnv',
    kwargs={
        'scenario': MultiAgentScenario.from_spec(
            path=f'{base_path}/../scenarios/austria.yml',
            rendering=True
        )
    })

register(id='MultiAgentAustria-v0',
    entry_point='racecar_gym.envs:MultiAgentRaceEnv',
    kwargs={
        'scenario': MultiAgentScenario.from_spec(
            path=f'{base_path}/../scenarios/austria.yml',
            rendering=False
        )
    })

register(id='MultiAgentBerlin_Gui-v0',
    entry_point='racecar_gym.envs:MultiAgentRaceEnv',
    kwargs={
        'scenario': MultiAgentScenario.from_spec(
            path=f'{base_path}/../scenarios/berlin.yml',
            rendering=True
        )
    })

register(id='MultiAgentBerlin-v0',
         entry_point='racecar_gym.envs:MultiAgentRaceEnv',
         kwargs={
             'scenario': MultiAgentScenario.from_spec(
                 path=f'{base_path}/../scenarios/berlin.yml',
                 rendering=False
             )
         })

register(id='MultiAgentTorino_Gui-v0',
         entry_point='racecar_gym.envs:MultiAgentRaceEnv',
         kwargs={
             'scenario': MultiAgentScenario.from_spec(
                 path=f'{base_path}/../scenarios/torino.yml',
                 rendering=True
             )
         })

register(id='MultiAgentTorino-v0',
         entry_point='racecar_gym.envs:MultiAgentRaceEnv',
         kwargs={
             'scenario': MultiAgentScenario.from_spec(
                 path=f'{base_path}/../scenarios/torino.yml',
                 rendering=False
             )
         })

register(id='MultiAgentMontreal_Gui-v0',
         entry_point='racecar_gym.envs:MultiAgentRaceEnv',
         kwargs={
             'scenario': MultiAgentScenario.from_spec(
                 path=f'{base_path}/../scenarios/montreal.yml',
                 rendering=True
             )
         })

register(id='MultiAgentMontreal-v0',
         entry_point='racecar_gym.envs:MultiAgentRaceEnv',
         kwargs={
             'scenario': MultiAgentScenario.from_spec(
                 path=f'{base_path}/../scenarios/montreal.yml',
                 rendering=False
             )
         })