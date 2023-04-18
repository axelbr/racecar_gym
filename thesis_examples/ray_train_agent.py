#trying to get a minimal example of training working with rllib

import numpy as np
import pprint
import random
import matplotlib.pyplot as plt

from ray.rllib.agents.ppo import PPOTrainer
from ray_wrapper import RayWrapper
import gymnasium
from racecar_gym.envs.gym_api import MultiAgentRaceEnv
from ray.tune.registry import register_env
from ray.rllib.algorithms import ppo
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from ray.rllib.policy.policy import PolicySpec
from dictionary_space_utility import flatten_obs
from time import sleep


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

#creating different policies for different agents...not sure if this is necessary since all agents have the same
#rewards and tasks
#also not very programatic
policies = {

    "A": PolicySpec(policy_class = None, observation_space = None, action_space = None, config = None),
    "B": PolicySpec(policy_class = None, observation_space = None, action_space = None, config = None),
    "C":PolicySpec(policy_class = None, observation_space = None, action_space = None, config = None),
    "D":PolicySpec(policy_class = None, observation_space = None, action_space = None, config = None)

}
config = PPOConfig()
config = config.environment(env = "my_env", env_config ={"num_agents": 4})
config = config.multi_agent(policies = policies,
                            policy_mapping_fn = policy_mapping_fn)



algo = config.build()

for _ in range(1):
    results = algo.train()
    print(pretty_print(results))





env = gymnasium.make(
        id='MultiAgentRaceEnv-v0',
        scenario='../scenarios/austria_het.yml',
        render_mode="rgb_array_follow",
        render_options=dict(width=320, height=240, agent='A')
    )

ray_env = RayWrapper(env)

#function for simulating agents with a trained model, also collect a dictionary of trajectories for the agent
def simulate(ray_env,algo,eps):
    joint_trajectories = {}
    policy_agent_mapping = algo.config['multiagent']['policy_mapping_fn']
    for episode in range(eps):
        trajectories = dict.fromkeys('Episode {}'.format(episode))
        print('Episode: {}'.format(episode))
        obs, _ = ray_env.reset()
        #print(obs)
        done = {agent: False for agent in obs.keys()}

        while True: # Run until the episode ends
            # Get actions from policies
            joint_action = {}
            for agent_id, agent_obs in obs.items():
                if done[agent_id]: continue # Don't get actions for done agents
                policy_id = policy_agent_mapping(agent_id,episode,None)
                action = algo.compute_single_action(agent_obs, policy_id=policy_id)
                joint_action[agent_id] = action



            # Step the simulation
            obs, reward, done, truncated, info = ray_env.step(joint_action)

            for agent_id in obs.keys():
                # might need to flatten this in the future --> nested dict structure
                trajectories['Episode {}'.format(episode)][agent_id]['action'] = joint_action[agent_id]
                trajectories['Episode {}'.format(episode)][agent_id]['pose'] = info[agent_id]['pose']







            #rgb_array = ray_env.render()
            #print(rgb_array.shape)
            #plt.clear()
            #plt.imshow(rgb_array)
            sleep(0.01)
            if done['__all__']:
                break
