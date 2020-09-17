import os

from gym.envs.registration import register

from racecar_gym.envs.scenarios import MultiAgentScenario

base_path = os.path.dirname(__file__)

register(id='f1tenth-berlin-two-gui-v0',
         entry_point='racecar_gym.envs.multi_race_car_env:MultiAgentRaceCarEnv',
         kwargs={
             'scenario': MultiAgentScenario.from_spec(f'{base_path}/../scenarios/berlin_two_agents_gui.yml')
         })
