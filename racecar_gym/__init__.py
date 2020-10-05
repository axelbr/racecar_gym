import os
from gym.envs.registration import register
from racecar_gym.envs.scenarios import MultiAgentScenario

base_path = os.path.dirname(__file__)

register(id='straight-gui-v0',
         entry_point='racecar_gym.envs.multi_race_car_env:MultiAgentRaceCarEnv',
         kwargs={
             'scenario': MultiAgentScenario.from_spec(
                 path=f'{base_path}/../scenarios/straight.yml',
                 rendering=True
             )
         })

register(id='austria-four-gui-v0',
         entry_point='racecar_gym.envs.multi_race_car_env:MultiAgentRaceCarEnv',
         kwargs={
             'scenario': MultiAgentScenario.from_spec(
                 path=f'{base_path}/../scenarios/austria.yml',
                 rendering=True
             )
         })

register(id='austria-four-v0',
         entry_point='racecar_gym.envs.multi_race_car_env:MultiAgentRaceCarEnv',
         kwargs={
             'scenario': MultiAgentScenario.from_spec(
                 path=f'{base_path}/../scenarios/austria.yml',
                 rendering=False
             )
         })