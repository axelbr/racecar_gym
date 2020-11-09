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

env: MultiAgentRaceEnv = gym.make('MultiAgentAustria_Gui-v0')
agent = GapFollower()

done = False
obs = env.reset(mode='random')
t = 0
while not done:
    action = env.action_space.sample()
    action_gf = agent.action(obs['A'])
    action['A'] = {'motor': action_gf[:2], 'steering': action_gf[-1]}
    obs, rewards, dones, states = env.step(action)
    done = any(dones.values())
    sleep(0.01)
    if t % 10 == 0:
        image = env.render(mode='follow', agent='A')
    t += 1


env.close()