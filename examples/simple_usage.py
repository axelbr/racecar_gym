from time import sleep
import gym
from racecar_gym.envs import MultiAgentRaceEnv

env = gym.make('SingleAgentAustria_Gui-v0')

done = False

# Currently, there are two reset modes available: 'grid' and 'random'.
# Grid: Place agents on predefined starting position.
# Random: Random poses on the track.
obs = env.reset(mode='grid')
t = 0
max_vel = 0
while not done:
    action = env.action_space.sample()
    action['steering'] = 0
    action['motor'] = 1.0
    if t > 250:
        print('BRAKE!')
        action['motor'] = -0.3
    obs, rewards, dones, states = env.step(action)
    #print(obs['velocity'])
    max_vel = max(max_vel, obs['velocity'][0])
    #print(max_vel)
    done = done#any(dones.values())
    sleep(0.01)
    if t % 100 == 0:
        # Currently, two rendering modes are available: 'birds_eye' and 'follow'
        # birds_eye: Follow an agent in birds eye perspective.
        # follow: Follow an agent in a 3rd person view.
        image = env.render(mode='follow')
    t+=1


env.close()
