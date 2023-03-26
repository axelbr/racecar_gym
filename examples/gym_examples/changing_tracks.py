from racecar_gym.envs import gym_api

scenarios = [f'../scenarios/{track}.yml' for track in ['austria', 'gbr', 'barcelona']]
env = gym_api.ChangingTrackSingleAgentRaceEnv(scenarios=scenarios, order='random')

for i in range(100):
    obs = env.reset()
    for _ in range(500):
        action = env.action_space.sample()
        _ = env.step(action)
env.close()