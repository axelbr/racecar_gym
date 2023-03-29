from racecar_gym.envs import pettingzoo_api

env = pettingzoo_api.env(scenario='../scenarios/austria.yml', render_mode='human')
obs = env.reset()
policy = lambda obs, agent: env.action_space(agent).sample()
done = False
while not done:
    action = dict((agent, policy(obs, agent)) for agent, obs in obs.items())
    observation, reward, done, truncated, info = env.step(action)
    done = all(done.values())
