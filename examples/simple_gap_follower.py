import time

import gym
from agents.gap_follower import GapFollower

from time import sleep
from racecar_gym import SingleAgentScenario, MultiAgentRaceEnv
from racecar_gym.envs.single_agent_race import SingleAgentRaceEnv
import numpy as np

from racecar_gym.tasks import Task, register_task
from racecar_gym.tasks.progress_based import MaximizeProgressTask, MaximizeContinuousProgressTask
import matplotlib.pyplot as plt

register_task(name='maximize_cont_progress', task=MaximizeContinuousProgressTask)

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
        self.env.render(mode)

scenario = SingleAgentScenario.from_spec(
                 path='custom.yml',
                 rendering=True
             )
env = SingleAgentRaceEnv(scenario=scenario)
env = SingleWrapper(env)
agent = GapFollower()

def run(env, agent):
    done = False
    obs = env.reset()

    init = time.time()
    reward_list = []
    progress_list = []
    progress_plus_list = []
    while not done:
        #agent_action = agent.action(obs)
        agent_action = env.action_space.sample()
        #print(agent_action)
        #print(random_action)
        #print()
        action_rewards = 0
        for _ in range(4):
            obs, rewards, done, states = env.step(agent_action)
            action_rewards += rewards
            env.render(mode='follow')
            if done:
                break
        print(f'Time: {states["time"]}, Lap: {states["lap"]}, Progress: {states["progress"]}, Reward: {action_rewards}')
        sleep(0.0005)
        reward_list.append(action_rewards)
        progress_list.append(states["progress"])
        progress_plus_list.append(states["lap"] + states["progress"])
    print("[Info] Track completed in {:.3f} seconds".format(time.time() - init))
    print("[Info] Return Value: {:.3f}".format(sum(reward_list)))
    return reward_list, progress_list, progress_plus_list

def plot_reward(reward_list, progress_list=None, progress_plus_list=None):
    if progress_list or progress_plus_list:
        plt.subplot(2, 1, 1)
    plt.plot(range(len(reward_list)), reward_list, label="reward")
    plt.legend()
    if progress_list or progress_plus_list:
        plt.subplot(2, 1, 2)
        if progress_list:
            plt.plot(range(len(progress_list)), progress_list, label="progress")
        else:
            plt.plot(range(len(progress_list)), progress_plus_list, label="progress +  lap")
        plt.legend()
    plt.show()


returns = []
for ep in range(50):
    rewards, progresses, prog_plus = run(env, agent)
    if len(rewards) > 1:
        pass
        #plot_reward(rewards, progresses, prog_plus)
    returns.append(sum(rewards))
env.close()

print("")
print(f"AVG RETURN: {sum(returns)/len(returns)}")
