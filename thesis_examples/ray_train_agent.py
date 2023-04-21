#trying to get a minimal example of training working with rllib

import numpy as np
import pprint
import random
import matplotlib.pyplot as plt
import cv2
import yaml
import wandb
import ray

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
from ray.rllib.algorithms.algorithm import Algorithm
from time import sleep
from collections import defaultdict




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



epochs = 100000
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
    #print(pretty_print(results))

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
        checkpoint = algo.save("/home/christine/trained_models")
        log_dict["checkpoint"] = checkpoint
    wandb.log(log_dict)




env = gymnasium.make(
        id='MultiAgentRaceEnv-v0',
        scenario='../scenarios/austria_het.yml',
        render_mode="human",
        #render_options=dict(width=320, height=240, agent='A')
    )

ray_env = RayWrapper(env)

#function for simulating agents with a trained model, also collect a dictionary of trajectories for the agent
def simulate(ray_env,algo,eps):
    policy_agent_mapping = algo.config['multiagent']['policy_mapping_fn']
    video_array = []
    trajectories = {}
    for episode in range(eps):
        trajectories['Episode{}'.format(episode)] = {}
        print('Episode: {}'.format(episode))
        obs, _ = ray_env.reset()
        #print(obs)
        done = {agent: False for agent in obs.keys()}

        img_array = []
        timesteps = []
        timestep = 0
        traj = {}
        for agent_id in obs.keys():
            traj['x_pos_{}'.format(agent_id)] = []
            traj['y_pos_{}'.format(agent_id)] = []
            traj['yaw_pos_{}'.format(agent_id)] = []

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
                traj['x_pos_{}'.format(agent_id)].append(info[agent_id]['pose'][0])
                traj['y_pos_{}'.format(agent_id)].append(info[agent_id]['pose'][1])
                traj['yaw_pos_{}'.format(agent_id)].append(info[agent_id]['pose'][5])

            timesteps.append(timestep)
            timestep = timestep + 1

            #rgb_array = ray_env.render()
            #ray_env.render()
            #transfer to BGR for openCV

            #img_array.append(cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR))

            #rgb_array = ray_env.render()
            #sleep(0.01)

            if done['__all__']:
                #video_array.append(img_array)
                trajectories['Episode{}'.format(episode)] = traj
                trajectories['Episode{}'.format(episode)]['timesteps'] = timesteps
                break
    return trajectories


checkpoint_path = "/home/christine/trained_models/checkpoint_001051"
algo = Algorithm.from_checkpoint(checkpoint_path)
#vids = simulate(ray_env,algo,1)
trajectories = simulate(ray_env,algo,10)


#assume the input is a set of images representing one video --> not working
def createvideo(vid):
    episode = 1
    filename = "trained_vid.mp4"
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 30, (320,240))
    for pic in vid:
        out.write(pic)

    out.release()

#takes a dictiionary containing trajectories and saves them to yml file
def save_trajs(trajectories,filepath = "test.yml"):
    file = open(filepath,"w")
    yaml.dump(trajectories,file)
    file.close()
    print("trajectories file saved")
