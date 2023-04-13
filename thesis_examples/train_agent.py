import sys
import gymnasium
sys.modules["gym"] = gymnasium

from gymnasium.vector.utils import concatenate, create_empty_array, iterate

import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from supersuit import color_reduction_v0, frame_stack_v1, resize_v1
from torch.distributions.categorical import Categorical

from stable_baselines3.common.utils import obs_as_tensor, safe_mean, configure_logger
from matplotlib import pyplot as plt

from stable_baselines3 import PPO, A2C, DQN

import supersuit as ss

import argparse

from racecar_gym.envs import pettingzoo_api
from stable_baselines3.ppo import CnnPolicy, MlpPolicy, MultiInputPolicy
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from flatten_action_wrapper import FlattenAction

from dictionary_space_utility import unwrap_obs_space, refit_obs


from sb3env_wrapper import SB3Wrapper


def batchify_obs(obs, device):
    """Converts PZ style observations to batch of torch arrays."""
    # convert to list of np arrays
    obs = np.stack([obs[a] for a in obs], axis=0)
    # # transpose to be (batch, channel, height, width)
    # obs = obs.transpose(0, -1, 1, 2)
    # convert to torch
    obs = torch.tensor(obs).to(device)

    return obs


def batchify(x, device):
    """Converts PZ style returns to batch of torch arrays."""
    # convert to list of np arrays
    x = np.stack([x[a] for a in x], axis=0)
    # convert to torch
    x = torch.tensor(x).to(device)

    return x


def unbatchify(x, env):
    """Converts np array to PZ style arguments."""
    # x = x.cpu().numpy()
    x = {a: x[i] for i, a in enumerate(env.possible_agents)}

    return x

def concat_obs(env,obs_dict):

    obs_list = []
    for i, agent in enumerate(env.par_env.possible_agents):
        if agent not in obs_dict:
            raise AssertionError(
                "environment has agent death. Not allowed for pettingzoo_env_to_vec_env_v1 unless black_death is True"
            )
        obs_list.append(obs_dict[agent])

    return concatenate(
        env.observation_space,
        obs_list,
        create_empty_array(env.observation_space, env.num_envs),
    )

def collate_dict(self, observation):
    if type(observation) is not dict:
        # assume tensor
        return observation
    res = []
    for obs, obs_value in observation.items():
        if type(obs_value) == dict:
            sub_obs = self.collate_dict(obs_value)
            obs_value = sub_obs
        elif type(obs_value) == list:
            obs_value = torch.Tensor(obs_value)
        elif type(obs_value) is int or type(obs_value) is float:
            obs_value = torch.Tensor([obs_value])
        res.append(obs_value)
    res = torch.cat(res, 0)
    return res

def run():

    #TODO(christine):
    # 1. Set up rest of params
    # 2. Get the environment to step forward (query policy + env.step(action))
    # 3. Add experiences to rollout buffer
    # 4. Make sure we're training


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ent_coef = 0.1
    vf_coef = 0.1
    clip_coef = 0.1
    gamma = 0.99
    batch_size = 32
    stack_size = 1
    obs_size = 60
    max_cycles = 125
    total_episodes = 100

    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="population-learning",
    # )

    parser = argparse.ArgumentParser(description=__doc__)

    # Activate verbose mode with --verbose=True.
    parser.add_argument(
        "--verbose", type=bool, default=False, help="Print debug information"
    )
    # Activate events printing mode with --print_events=True.

    parser.add_argument("--rollout-len", type=int, default=1000)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--num-frames", type=int, default=1)
    parser.add_argument("--render", type=bool, default=False)
    parser.add_argument(
        "--agent_color_pop_0", action="store", default=np.array([25, 200, 255]) / 255
    )
    parser.add_argument(
        "--agent_color_pop_1", action="store", default=np.array([35, 155, 0]) / 255
    )
    parser.add_argument("--agent_size", type=float, default=0.1)
    parser.add_argument("--landmark_size", type=float, default=0.05)
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=10,
        help="Number of episodes between evaluations",
    )

    args = parser.parse_args()
    params = vars(args)

    # TODO(justin.lidard): add render from command line args
    render = True
    if render:
        params["render"] = True

    num_frames = params["num_frames"]
    rollout_len = params["rollout_len"]
    eval_rollout_len = rollout_len
    num_episodes = params["num_episodes"]
    eval_interval = params["eval_interval"]

    """ ENV SETUP """

    # env = simple_spread_v3.parallel_env(
    #     render_mode="rgb_array",
    #     continuous_actions=False,
    #     max_cycles=max_cycles,
    #     N=10,
    #     params=params,
    # )
    #env = pettingzoo_api.env(scenario='../scenarios/austria_het.yml', render_mode="rgb_array_follow")

    #env = gymnasium.make('SingleAgentAustria-v0', render_mode='rgb_array_follow')
    #env = gymnasium.wrappers.FlattenObservation(env)

    env = gymnasium.make(
        id='MultiAgentRaceEnv-v0',
        scenario='../scenarios/austria_het.yml',
        render_mode="rgb_array_follow"
    )

    #env = pettingzoo_api.env(scenario='../scenarios/austria_het.yml', render_mode="rgb_array_follow")

    #env = ss.pettingzoo_env_to_vec_env_v1(env)


    #env = FlattenAction(env)
    num_agents = len(env.possible_agents)

    #seems like flattening the observations looses information?? or the order of info is scrambled
    #env = gymnasium.wrappers.FlattenObservation(env)


    #environment wrapper for handling sb3
    sb3_env = SB3Wrapper(env)
    parallel_env = env

    #print(sb3_env.observation_space)






    baseline = "PPO"
    if baseline == "PPO":
        agent_model = PPO("MultiInputPolicy", sb3_env)
    elif baseline == "A2C":
        agent_model = A2C("MlpPolicy", parallel_env)
    elif baseline == "DQN":
        agent_model = DQN("MlpPolicy", parallel_env)
    else:
        raise NotImplementedError("Baseline Strategy not Implemented.")


    # env = color_reduction_v0(env)
    # env = resize_v1(env, frame_size[0], frame_size[1])
    # env = frame_stack_v1(env, stack_size=stack_size)
    # TODO(christine): make sure the rest of the parameters are set up
    #num_agents = len(parallel_env.par_env.possible_agents) # look up how to do this programatically
    #print(num_agents)

    #not sure if this is necessary for a continuous action space??
    #num_actions = parallel_env.par_env.action_space(parallel_env.par_env.possible_agents[0]).n

    #not sure if this is needed, come back later. Observation size might be a bit tricky due to dict structure
    #observation_size = parallel_env.par_env.observation_space.shape


    """ For Plotting Purposes """
    eps = []
    ep_return = []
    val_loss = []
    pol_loss = []

    validation = False

    agent_model._logger = configure_logger(0)

    """ TRAINING LOGIC """
    # train for n number of episodes
    for episode in range(total_episodes):
        print(f"episode:{episode}")
        log_dict = {}

        train = episode % eval_interval != 0
        eval = episode % eval_interval == 0

        if validation:
            rollout_len = eval_rollout_len
            training_mode_prefix = "validation"
        else:
            rollout_len = params["rollout_len"]
            training_mode_prefix = "train"

        agent_rewards = {}
        for agent in parallel_env.possible_agents:
            agent_rewards[agent] = 0

        #come back to this...
        total_episodic_return = 0

        # collect an episode
        with torch.no_grad():
            # collect observations and convert to batch of torch tensors

            #print(env.observation_space)
            #print(parallel_env.observation_space.spaces['A'])


            #print(parallel_env.observation_space)


            next_obs,state = parallel_env.reset(seed=None) #original obs space





            # reset the episodic return
            total_episodic_return = {}
            for agent in range(num_agents):
                total_episodic_return[agent] = 0

            # each episode has num_steps
            for t_step in range(rollout_len):
                if render:
                    parallel_env.render()

                #TODO(christine): make sure environment steps properly

                #this will not work with racecar environment
                #obs = batchify_obs(next_obs, device)

                # get action from the agent

                #collecting all the individual agents actions
                ind_actions = {}
                for key in sb3_env.possible_agents:
                    print(next_obs[key])
                    sb3_obs, _ = refit_obs(next_obs[key])
                    print(sb3_obs)

                    #error with .policy() function that is caused by the dimensions of the new observations (I think)
                    #actions, values, log_probs = agent_model.policy(sb3_obs)
                    actions = agent_model.predict(sb3_obs)

                    ind_actions[key] = actions

                    #After collecting all the individual actions, I now need to collate/combine all the actions into one action space to use with env.step()



                # check this --> probably not going to work well with the new action space and obs space
                # execute the environment and log data
                next_obs, rewards, terms, truncs, infos = parallel_env.step(
                    unbatchify(actions.numpy(), env)
                )

                for agent in env.possible_agents:
                    agent_rewards[agent] += rewards[agent]

                # compute episodic return
                total_episodic_return[agent] += rewards[agent]

                # Optimization step
                if train and t_step > 0:
                    episode_start = t_step == 1

                    # Collect transitions for each agent
                    # obs_batched = (
                    #     obs#obs_as_tensor(obs, device="cpu")
                    #     .unsqueeze(0)
                    #     .repeat(num_agents, 1, 1, 1)
                    # )

                    #TODO(christine): Add experiences to rollout buffer
                    obs_batched = obs
                    policy_outputs_batched = actions  # obs_as_tensor(policy_outputs, device="cpu").unsqueeze(0)
                    rewards_batched = torch.Tensor([float(x) for x in rewards.values()])
                    values_batched = (
                        values.detach()
                    )  # obs_as_tensor(values, device="cpu").unsqueeze(0)
                    log_probs_batched = log_probs.detach()

                    # Add to the replay buffer
                    if not agent_model.rollout_buffer.full:
                        agent_model.rollout_buffer.add(
                            obs_batched,
                            policy_outputs_batched,
                            rewards_batched,
                            episode_start,
                            values_batched,
                            log_probs_batched,
                        )

                    # Compute value for the last timestep
                    values_batched = agent_model.policy.predict_values(obs_batched)

        # TODO(christine): Make sure the agents are training with no error!
        # After episode terminates and we have enough data, do some learning
        if agent_model.rollout_buffer.full and train:
            agent_model.rollout_buffer.compute_returns_and_advantage(
                last_values=values_batched, dones=episode_start
            )
            agent_model.train()
            agent_model.rollout_buffer.reset()
        elif eval:
            log_dict[f"{training_mode_prefix}_evaluation_episode_length"] = t_step

        # Logging
        for agent in env.possible_agents:
            log_dict.update({f"Agent Rewards/{agent}": agent_rewards[agent]})
        log_dict.update(
            {
                "Total Episodic Return/Total Episodic Return": np.mean(
                    total_episodic_return
                )
            }
        )


if __name__ == "__main__":
    run()