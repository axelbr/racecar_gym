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
while not done:
    agent_action = agent.action(obs)
    #agent_action = env.action_space.sample()
    #print(agent_action)
    #print(random_action)
    #print()
    obs, rewards, done, states = env.step(agent_action)
    if last_progress < 0:
        last_progress = states["progress"]
    delta_progress = states["progress"] - last_progress
    r = 0
    if delta_progress > 0:
        last_progress = states["progress"]
        r = progress_mult * delta_progress
    print(f'Time: {states["time"]}, Lap: {states["lap"]}, Progress: {states["progress"]}, Reward: {r}')
    sleep(0.0005)
    ret += r
print("[Info] Track completed in {:.3f} seconds".format(time.time() - init))
print("[Info] Return Value: {:.3f}".format(ret))
env.close()
