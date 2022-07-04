from racecar_gym.envs import pettingzoo_api
from pettingzoo.test import api_test

env = pettingzoo_api.raw_env(scenario_path='./austria.yml')
api_test(env, num_cycles=2000, verbose_progress=True)