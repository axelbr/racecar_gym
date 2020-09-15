import os

from gym.envs.registration import register

from racecar_gym.envs import load_spec, load_from_spec

base_path = os.path.dirname(__file__)

if True:
    register(id='f1tenth-berlin-two-v0',
             entry_point='racecar_gym.envs:load_from_spec',
             kwargs={'path': f'{base_path}/../scenarios/berlin_two_agents.yml'})

    register(id='f1tenth-berlin-two-gui-v0',
             entry_point='racecar_gym.envs:load_from_spec',
             kwargs={'path': f'{base_path}/../scenarios/porto_two_agents_gui.yml'})

    register(id='f1tenth-porto-two-gui-v0',
             entry_point='racecar_gym.envs:load_from_spec',
             kwargs={'path': f'{base_path}/../scenarios/porto_two_agents_gui.yml'})

    register(id='f1tenth-porto-two-v0',
             entry_point='racecar_gym.envs:load_from_spec',
             kwargs={'path': f'{base_path}/../scenarios/porto_two_agents.yml'})
