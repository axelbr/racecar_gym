from time import time, sleep

import gym
import racecar_gym
from agents.gap_follower import GapFollower

env = gym.make('austria-four-v0')
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

while not done:
    actions = {}
    for id, obs in observation.items():
        action = agent.action(obs)
        actions[id] = {'motor': (action[0], action[1]), 'steering': action[2]}
    observation, reward, dones, info = monitor_env.step(actions)
    # images.append(observation[1]['rgb_camera'])
    for id, reward in reward.items():
        rewards[id] += reward

    done = any(dones.values())
    print(rewards)
    i += 1
    #sleep(0.01)


#imageio.mimsave('movie.gif', images)
end = time()
print('wall time: ' + str((end-start)))
print('sim time: ' + str(i/100))
print('RTF: ', (i/100) / (end-start))
monitor_env.close()
env.close()
