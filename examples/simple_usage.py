from time import sleep
import gym
import racecar_gym

env = gym.make('MultiAgentBerlin_Gui-v0')

done = False
obs = env.reset()

while not done:
    action = env.action_space.sample()
    obs, rewards, dones, states = env.step(action)
    done = any(dones.values())
    sleep(0.01)

env.close()
