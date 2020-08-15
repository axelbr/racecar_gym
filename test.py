import math

import gym  # open ai gym
#import pybulletgym  # register PyBullet enviroments with open ai gym
from pybullet_envs.bullet.racecarGymEnv import RacecarGymEnv
import time
import numpy as np

from agents.gap_follower import GapFollower
from racecar_gym.single_car_env import SingleCarEnv, SingleCarEnvScenario
from race_env import RaceEnv
x = './m'
scenario = SingleCarEnvScenario(map='models/tracks/barca_track.sdf',
                                car='models/cars/racecar_differential.urdf',
                                initial_pose=np.array([0, 0, 0.1]))
env = SingleCarEnv(scenario=scenario)
env.render() # call this before racecar_gym.reset, if you want a window showing the environment
observation = env.reset()  # should return a state vector if everything worked
print(env.observation_space)
agent = GapFollower()
while True:
    action = agent.action(observation)
    observation, reward, done, info = env.step(action)
    print(observation)
    env.render()
