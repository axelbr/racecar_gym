from time import time, sleep

import numpy as np

from agents.gap_follower import GapFollower
import gym
from gym import wrappers
import racecar_gym

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
while not done:
    actions = [agent.action(obs) for obs in observation]
    observation, reward, dones, info = monitor_env.step(actions)
    rewards += np.array(reward)
    done = any(dones)
    i += 1

    print(rewards)
end = time()

print('wall time: ' + str((end-start)))
print('sim time: ' + str(i/100))
print('RTF: ', (i/100) / (end-start))
monitor_env.close()
env.close()
