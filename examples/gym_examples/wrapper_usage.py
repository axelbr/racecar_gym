from time import sleep
import gym
from racecar_gym.envs import gym_api

scenario = gym_api.SingleAgentScenario.from_spec(
    path='../scenarios/custom.yml',
    rendering=True
)

env = gym_api.VectorizedSingleAgentRaceEnv(scenarios=[scenario, scenario])
env = gym_api.wrappers.VectorizedSingleAgentActionRepeat(env, steps=4)


print(env.observation_space)
print(env.action_space)
done = False

# Currently, there are two reset modes available: 'grid' and 'random'.
# Grid: Place agents on predefined starting position.
# Random: Random poses on the track.
obs = env.reset(mode='grid')
t = 0
while not done:
    action = env.action_space.sample()
    obs, rewards, done, states = env.step(action)
    sleep(0.01)
    if t % 30 == 0:
        # Currently, two rendering modes are available: 'birds_eye' and 'follow'
        # birds_eye: Follow an agent in birds eye perspective.
        # follow: Follow an agent in a 3rd person view.
        image = env.render(mode='follow')
    t+=1
    done = all(done)
    print(rewards)


env.close()
