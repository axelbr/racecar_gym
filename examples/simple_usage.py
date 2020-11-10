from time import sleep
import gym
from racecar_gym.envs import MultiAgentRaceEnv

env: MultiAgentRaceEnv = gym.make('MultiAgentBerlin_Gui-v0')

done = False
obs = env.reset(mode='grid')
t = 0
while not done:
    action = env.action_space.sample()
    obs, rewards, dones, states = env.step(action)
    done = any(dones.values())
    sleep(0.01)
    if t % 10 == 0:
        image = env.render(mode='birds_eye', agent='A')
    t+=1


env.close()
