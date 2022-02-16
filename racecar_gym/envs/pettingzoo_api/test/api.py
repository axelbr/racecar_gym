from pettingzoo.test import api_test

from racecar_gym import MultiAgentScenario
from racecar_gym.envs.pettingzoo import racecarenv

scenario = MultiAgentScenario.from_spec(path='./austria.yml', rendering=True)
env = racecarenv.raw_env(scenario=scenario)
api_test(env, num_cycles=2000, verbose_progress=True)
