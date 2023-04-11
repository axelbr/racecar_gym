
#import sys


import gymnasium
#import sys
#sys.modules["gym"] = gym

from racecar_gym.envs import pettingzoo_api
from stable_baselines3.ppo import CnnPolicy, MlpPolicy, MultiInputPolicy
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env


#careful with versioning and supersuit, needs pettingzoo == 1.22.4
#SuperSuit==3.7.2

from pettingzoo.test import parallel_api_test
from pettingzoo.utils.conversions import aec_to_parallel
from pettingzoo.utils.conversions import parallel_to_aec

env = pettingzoo_api.env(scenario='../scenarios/austria_het.yml',render_mode="rgb_array_follow")

#print(env.render_mode)

#Add a feature/function to flatten action spaces


#parallel_api_test(env)

#env = pettingzoo_api.flatten(env)
#trying to get env setup to work with supersuit for multi-agent parameter sharing
env = ss.pettingzoo_env_to_vec_env_v1(env)

#parallel environment

#env = ss.flatten_v0(env)

#I get some sort of error when I try to parallelzie the training with multiple cpus?
#env = ss.concat_vec_envs_v1(env, 8, num_cpus=1, base_class='stable_baselines3')

#flattening the action space --> a bit hacky...
env.action_space = gymnasium.spaces.utils.flatten_space(env.action_space)

#print(env.observation_space)

#check_env(env)

env = gymnasium.wrappers.FlattenObservation(env)



#print(env.observation_space)




#env.observation_space = gym.spaces.dict.Dict()
#now the action space is a Box
#print(env.observation_space)


#training model with PPO using stable baselines
#stable baselines 3 does not support Dict action_spaces
model = PPO(MlpPolicy, env, verbose=2, gamma=0.95, n_steps=20, ent_coef=0.0905168, learning_rate=0.00062211,
            vf_coef=0.042202, max_grad_norm=0.9, gae_lambda=0.99, n_epochs=5, clip_range=0.3, batch_size=20)

#timesteps argument in the .learn() method refers to actions taken by an individual agent, not the total number of times the game is played.
model.learn(total_timesteps=10,progress_bar = True)



#obs = env.reset()
#policy = lambda obs, agent: env.action_space(agent).sample()
#done = False
#while not done:
#    action = dict((agent, policy(obs, agent)) for agent, obs in obs.items())
#    observation, reward, done, truncated, info = env.step(action)
#    done = all(done.values())
