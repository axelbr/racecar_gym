from time import sleep
import gymnasium
import racecar_gym.envs.gym_api

# Currently, three rendering modes are available: 'human', 'rgb_array_birds_eye' and 'rgb_array_follow'
# human: Render the scene in a window.
# rgb_array_birds_eye: Follow an agent in birds eye perspective.
# rgb_array_follow: Follow an agent in a 3rd person view.
env = gymnasium.make('SingleAgentAustria-v0', render_mode='rgb_array_follow')
done = False

# Currently, there are two reset modes available: 'grid' and 'random'.
# Grid: Place agents on predefined starting position.
# Random: Random poses on the track.
obs = env.reset(options=dict(mode='grid'))
t = 0
while not done:
    action = env.action_space.sample()
    obs, rewards, done, truncated, states = env.step(action)
    sleep(0.01)
    if t % 30 == 0:
        image = env.render()
    t+=1


env.close()
