from time import sleep
from racecar_gym.envs import gym_api

scenario = '../scenarios/custom.yml'

env = gym_api.VectorizedSingleAgentRaceEnv(scenarios=[scenario, scenario], render_mode='human')
env = gym_api.wrappers.VectorizedSingleAgentActionRepeat(env, steps=4)


print(env.observation_space)
print(env.action_space)
done = False

# Currently, there are two reset modes available: 'grid' and 'random'.
# Grid: Place agents on predefined starting position.
# Random: Random poses on the track.
obs = env.reset(options=dict(mode='grid'))
t = 0
while not done:
    action = env.action_space.sample()
    obs, rewards, terminated, truncated, states = env.step(action)
    sleep(0.01)
    if t % 30 == 0:
        image = env.render()
    t+=1
    done = all(terminated)
    print(rewards)


env.close()
