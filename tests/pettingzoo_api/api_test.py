from racecar_gym.envs import pz_envs
from pettingzoo.test import api_test

env = pz_envs.racecarenv.raw_env(scenario_path='./austria.yml')
api_test(env, num_cycles=2000, verbose_progress=True)