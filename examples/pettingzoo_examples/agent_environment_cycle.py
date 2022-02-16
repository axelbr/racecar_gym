from racecar_gym.envs import pz_envs

env = pz_envs.racecarenv.env(scenario_path='../scenarios/austria.yml', live_rendering=True)

env.reset()
policy = lambda obs, agent: env.action_space(agent).sample()
for agent in env.agent_iter(max_iter=2000):
    observation, reward, done, info = env.last()
    action = policy(observation, agent)
    env.step(action)
