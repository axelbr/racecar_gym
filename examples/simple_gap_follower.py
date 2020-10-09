import time

import gym
from agents.gap_follower import GapFollower

from time import sleep
from racecar_gym import SingleAgentScenario
from racecar_gym.envs.single_race_car_env import SingleAgentRaceCarEnv
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
                 path='custom_single_car.yml',
                 rendering=True
             )
env = SingleAgentRaceCarEnv(scenario=scenario)
env = SingleWrapper(env)
agent = GapFollower()

done = False
obs = env.reset()

init = time.time()
while not done:
    agent_action = agent.action(obs)
    #random_action = env.action_space.sample()
    #print(agent_action)
    #print(random_action)
    #print()
    obs, rewards, done, states = env.step(agent_action)
    sleep(0.0001)
print("[Info] Track completed in {:.3f} seconds".format(time.time() - init))
env.close()
