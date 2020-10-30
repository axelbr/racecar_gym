import time

import gym
from agents.gap_follower import GapFollower

from time import sleep
from racecar_gym import SingleAgentScenario
from racecar_gym.envs.single_agent_race import SingleAgentRaceEnv
import numpy as np

class SingleWrapper(gym.Env):
    def __init__(self, env):
        self.env = env
        self.action_space = gym.spaces.Box(
            np.append([2, 0.5], env.action_space['steering'].low),
            np.append(env.action_space['motor'].high, env.action_space['steering'].high))
        self.observation_space = env.observation_space['lidar']

    def step(self, action):
        action = {'motor': (action[0], action[1]), 'steering': action[2]}
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        return obs

    def render(self, mode='human'):
        pass

scenario = SingleAgentScenario.from_spec(
                 path='custom.yml',
                 rendering=True
             )
env = SingleAgentRaceEnv(scenario=scenario)
env = SingleWrapper(env)
agent = GapFollower()

done = False
obs = env.reset()

init = time.time()
last_progress = -1
progress_mult = 100
ret = 0
reward_list = []
progress_list = []
while not done:
    agent_action = agent.action(obs)
    #agent_action = env.action_space.sample()
    #print(agent_action)
    #print(random_action)
    #print()
    obs, rewards, done, states = env.step(agent_action)
    print(f'Time: {states["time"]}, Lap: {states["lap"]}, Progress: {states["progress"]}, Reward: {rewards}')
    sleep(0.0005)
    reward_list.append(rewards)
    progress_list.append(states["progress"])
print("[Info] Track completed in {:.3f} seconds".format(time.time() - init))
print("[Info] Return Value: {:.3f}".format(sum(reward_list)))

import matplotlib.pyplot as plt
plt.subplot(2, 1, 1)
plt.plot(range(len(reward_list)), reward_list, label="reward")
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(range(len(progress_list)), progress_list, label="progress")
plt.legend()
plt.show()
env.close()
