#trying to get a minimal example of training working with rllib

import numpy as np
import pprint
import random

from ray.rllib.agents.ppo import PPOTrainer
from ray_wrapper import RayWrapper
import gymnasium
from racecar_gym.envs.gym_api import MultiAgentRaceEnv
from ray.tune.registry import register_env
from ray.rllib.algorithms import ppo
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from ray.rllib.policy.policy import PolicySpec
import wandb


import ray


def env_creator(env_config):
    env = gymnasium.make(
        id='MultiAgentRaceEnv-v0',
        scenario='../scenarios/austria_het.yml',
        render_mode="rgb_array_follow"
    )

    return RayWrapper(env)

#maps policies to agents...I don't fully understand this feature

def policy_mapping_fn(agent_id,episode,worker,**kwargs):
    pol_id = agent_id
    return pol_id

register_env("my_env",env_creator)


#ray.init()
#algo = PPOTrainer(env="my_env", config = {
#    "multiagent":{}, "env")
#results = algo.train()

epochs = 10
rollout_fragment_length = 1000
params = {"epochs": epochs,
          "rollout_fragment_length": rollout_fragment_length} #TODO(christine.ohenzuwa): add command line args using python argparse
# https://docs.python.org/3/library/argparse.html


wandb.init(
    # Set the project where this run will be logged
    project="population-learning",
    # Track hyperparameters and run metadata
    config=params)

#creating different policies for different agents...not sure if this is necessary since all agents have the same
#rewards and tasks
#also not very programatic
policies = {

    "A": PolicySpec(policy_class = None, observation_space = None, action_space = None, config = None),
    "B": PolicySpec(policy_class = None, observation_space = None, action_space = None, config = None),
    "C":PolicySpec(policy_class = None, observation_space = None, action_space = None, config = None),
    "D":PolicySpec(policy_class = None, observation_space = None, action_space = None, config = None)

}
config = PPOConfig().framework("torch").rollouts(rollout_fragment_length=params["rollout_fragment_length"])
config = config.environment(env = "my_env", env_config ={"num_agents": 4})
config = config.multi_agent(policies = policies,
                            policy_mapping_fn = policy_mapping_fn)



algo = config.build()
checkpoint = 10
for epoch in range(params["epochs"]):

    # Train the model for (1?) epoch with a pre-specified rollout fragment length (how many rollouts?).
    results = algo.train()
    print(pretty_print(results))

    # Log the results
    log_dict = {}  # reset the log dict
    results_top_level_stats = {"stats/" + k:v for (k,v) in results.items() if k != "info" and type(v) is not dict}
    results_info_stats = {"info/" + k:v for (k,v) in results["info"].items() if k != "learner" and type(v) is not dict}
    for d in [results_info_stats, results_top_level_stats]:
        log_dict.update(d)
    for agent_prefix, agent_dict in results["info"]["learner"].items():
        learner_stats = {"learner_agent_" + agent_prefix + "/" + k:v for (k,v) in agent_dict["learner_stats"].items()}
        log_dict.update(learner_stats)
    if epoch % 10 == 0:
        checkpoint = algo.save()
        log_dict["checkpoint"] = checkpoint
    wandb.log(log_dict)

