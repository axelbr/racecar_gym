from time import time, sleep

import numpy as np

from agents.gap_follower import GapFollower
import gym
from gym import wrappers
import racecar_gym
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import imageio

env = gym.make('f1tenth-berlin-two-v0')
monitor_env = env#wrappers.Monitor(env, directory='../recordings', force=True, video_callable=lambda episode_id: True)
#env.render()
observation = monitor_env.reset()
agent = GapFollower()
done = False

print(env.observation_space)

i = 0
start = time()
rewards = np.zeros(shape=len(observation))
images = []

while not done:
    actions = [agent.action(obs) for obs in observation]
    observation, reward, dones, info = monitor_env.step(actions)
    #images.append(observation[1]['rgb_camera'])
    rewards += np.array(reward)
    done = any(dones)
    i += 1
    print(rewards)

#imageio.mimsave('movie.gif', images)
end = time()
print('wall time: ' + str((end-start)))
print('sim time: ' + str(i/100))
print('RTF: ', (i/100) / (end-start))
monitor_env.close()
env.close()
