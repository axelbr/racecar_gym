from racecar_gym.envs import pettingzoo_api
from pettingzoo.test import parallel_api_test

parallel_env = pettingzoo_api.parallel_env(scenario_path='./austria.yml')
parallel_api_test(parallel_env, num_cycles=2000)