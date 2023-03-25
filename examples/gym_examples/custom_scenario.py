from time import sleep
import gymnasium
from racecar_gym.envs import gym_api

env = gymnasium.make(
    id='MultiAgentRaceEnv-v0',
    scenario='../scenarios/custom.yml',
    render_mode='human'
)

print(env.observation_space)
print(env.action_space)

done = False
obs = env.reset(options=dict(mode='grid'))

while not done:
    action = env.action_space.sample()
    obs, rewards, dones, truncated, states = env.step(action)
    done = any(dones.values())
    env.render()
    sleep(0.01)

env.close()
