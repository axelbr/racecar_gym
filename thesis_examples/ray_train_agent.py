#trying to get a minimal example of training working with rllib

import numpy as np
import pprint
from ray.rllib.agents.ppo import PPOTrainer
from ray_wrapper import RayWrapper
import gymnasium
from racecar_gym.envs.gym_api import MultiAgentRaceEnv
from ray.tune.registry import register_env
from ray.rllib.algorithms import ppo
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print

import ray


def env_creator(env_config):
    env = gymnasium.make(
        id='MultiAgentRaceEnv-v0',
        scenario='../scenarios/austria_het.yml',
        render_mode="rgb_array_follow"
    )

    return RayWrapper(env)

register_env("my_env",env_creator)


#ray.init()
#algo = PPOTrainer(env="my_env", config = {
#    "multiagent":{}, "env")
#results = algo.train()

config = PPOConfig()
config = config.environment(env = "my_env", env_config ={"num_agents": 4})
algo = config.build()

for _ in range(10):
    results = algo.train()
    print(pretty_print(results))

