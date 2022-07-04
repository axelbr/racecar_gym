from racecar_gym.envs import gym_api


n_parallel_instances = 2
rendering = True
scenarios = [gym_api.SingleAgentScenario.from_spec('../scenarios/custom.yml', rendering=rendering) for _ in range(n_parallel_instances)]
env = gym_api.VectorizedSingleAgentRaceEnv(scenarios=scenarios)
n_agents_per_instance = [len(act.spaces.keys()) for act in env.action_space]

for i in range(3):
    done = False
    obs = env.reset()
    episode = []
    while not done:
        action = env.action_space.sample()
        obs, rewards, dones, states = env.step(action)
        done = any([e for e in dones])
        renderings = env.render()
        episode.append(obs)
env.close()