from time import time, sleep

from agents.gap_follower import GapFollower
import gym
from gym import wrappers
import racecar_gym

env = gym.make('f1tenth-porto-two-gui-v0')
monitor_env = env#wrappers.Monitor(env, directory='../recordings', force=True, video_callable=lambda episode_id: True)
#env.render()
observation = monitor_env.reset()
agent = GapFollower()
done = False
i = 0
start = time()
while not done:
    actions = [agent.action(obs) for obs in observation]
    observation, reward, dones, info = monitor_env.step(actions)
    done = any(dones)
    i += 1
    #sleep(0.01)
end = time()

print('wall time: ' + str((end-start)))
print('sim time: ' + str(i/100))
print('RTF: ', (i/100) / (end-start))
monitor_env.close()
env.close()
