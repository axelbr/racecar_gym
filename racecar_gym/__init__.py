import os

from gym.envs.registration import register

from racecar_gym.envs.scenarios import MultiAgentScenario

base_path = os.path.dirname(__file__)

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

register(id='f1tenth-berlin-two-gui-v0',
         entry_point='racecar_gym.envs.multi_race_car_env:MultiAgentRaceCarEnv',
         kwargs={
             'scenario': MultiAgentScenario.from_spec(f'{base_path}/../scenarios/berlin_two_agents_gui.yml')
         })

register(id='f1tenth-berlin-two-v0',
         entry_point='racecar_gym.envs.multi_race_car_env:MultiAgentRaceCarEnv',
         kwargs={
             'scenario': MultiAgentScenario.from_spec(f'{base_path}/../scenarios/berlin_two_agents.yml')
         })

register(id='f1tenth-porto-two-gui-v0',
         entry_point='racecar_gym.envs.multi_race_car_env:MultiAgentRaceCarEnv',
         kwargs={
             'scenario': MultiAgentScenario.from_spec(f'{base_path}/../scenarios/porto_two_agents_gui.yml')
         })

register(id='f1tenth-porto-two-v0',
         entry_point='racecar_gym.envs.multi_race_car_env:MultiAgentRaceCarEnv',
         kwargs={
             'scenario': MultiAgentScenario.from_spec(f'{base_path}/../scenarios/porto_two_agents.yml')
         })
