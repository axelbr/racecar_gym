from racecar_gym.envs import gym_api
from racecar_gym import MultiAgentScenario, SingleAgentScenario

scenarios = [f'../scenarios/{track}.yml' for track in ['austria', 'barcelona']]
env = gym_api.ChangingTrackMultiAgentRaceEnv(scenarios=scenarios, order='manual', render_mode='human')

for _ in range(4):
    env.set_next_env()
    for i in range(2):
        obs = env.reset()
        for _ in range(500):
            action = env.action_space.sample()
            _ = env.step(action)
            env.render()

env.close()