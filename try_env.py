from functools import partial
from time import time, sleep

import gym
import numpy as np
import racecar_gym
from agents.gap_follower import GapFollower

class SingleWrapper(gym.Env):

    def __init__(self, env):
        self.env = env
        self.action_space = gym.spaces.Box(
            np.append([2, 0.5], env.action_space['A']['steering'].low),
            np.append(env.action_space['A']['motor'].high, env.action_space['A']['steering'].high))
        self.observation_space = env.observation_space['A']['lidar']

    def step(self, action):
        action = {'motor': (action[0], action[1]), 'steering': action[2]}
        obs, reward, done, info = self.env.step({'A': action})
        return obs['A']['lidar'], reward['A'], done['A'], info['A']

    def reset(self):
        obs = self.env.reset()
        return obs['A']['lidar']

    def render(self, mode='human'):
        pass

env = gym.make('MultiAgentTrack1_Gui-v0')
monitor_env = env  # wrappers.Monitor(env, directory='../recordings', force=True, video_callable=lambda episode_id: True)
#env.render()
observation = monitor_env.reset()
agent = GapFollower()
done = False

print(env.observation_space)
print(env.action_space)

i = 0
start = time()
rewards = dict([(id, 0) for id in observation.keys()])
images = []

env = SingleWrapper(env)

