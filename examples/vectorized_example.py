from racecar_gym import MultiAgentScenario
from racecar_gym.envs.vectorized_multi_agent_race import VectorizedMultiAgentRaceEnv
from agents.gap_follower import GapFollower

n_parallel_instances = 2
rendering = True
scenarios = [MultiAgentScenario.from_spec('custom.yml', rendering=rendering) for _ in range(n_parallel_instances)]
env = VectorizedMultiAgentRaceEnv(scenarios=scenarios)
n_agents_per_instance = [len(act.spaces.keys()) for act in env.action_space]
gfollow = GapFollower()

for i in range(3):
    done = False
    obs = env.reset()
    episode = []
    while not done:
        action = []
        for i in range(n_parallel_instances):
            multi_action = dict()
            for agent_id in obs[i].keys():
                act = gfollow.action(obs[i][agent_id])
                act = {'motor': act[0], 'steering': act[1]}
                multi_action[agent_id] = act
            action.append(multi_action)
        #action = env.action_space.sample()
        obs, rewards, dones, states = env.step(action)
        done = any([any(e.values()) for e in dones])
        renderings = env.render()
        episode.append(obs)
env.close()