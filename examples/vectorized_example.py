from racecar_gym import MultiAgentScenario, VectorizedSingleAgentRaceEnv, SingleAgentScenario
from racecar_gym.envs.vectorized_multi_agent_race import VectorizedMultiAgentRaceEnv
from agents.gap_follower import GapFollower

n_parallel_instances = 2
rendering = True
scenarios = [SingleAgentScenario.from_spec('scenarios/custom.yml', rendering=rendering) for _ in range(n_parallel_instances)]
env = VectorizedSingleAgentRaceEnv(scenarios=scenarios)
n_agents_per_instance = [len(act.spaces.keys()) for act in env.action_space]
gfollow = GapFollower()

for i in range(3):
    done = False
    obs = env.reset()
    episode = []
    while not done:
        action = env.action_space.sample()
        #action = env.action_space.sample()
        obs, rewards, dones, states = env.step(action)
        done = any([e for e in dones])
        renderings = env.render()
        episode.append(obs)
env.close()