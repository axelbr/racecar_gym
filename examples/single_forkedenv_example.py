from racecar_gym import MultiAgentScenario
from racecar_gym.envs.forked_multi_agent_race import ForkedMultiAgentRaceEnv
from agents.gap_follower import GapFollower
import matplotlib.pyplot as plt
import numpy as np

n_parallel_instances = 2
rendering = False
scenarios = [MultiAgentScenario.from_spec('custom1.yml', rendering=rendering),
             MultiAgentScenario.from_spec('custom2.yml', rendering=rendering)]
train_env = ForkedMultiAgentRaceEnv(scenario=scenarios[0])
test_env = ForkedMultiAgentRaceEnv(scenario=scenarios[1])
envs = [train_env, test_env]

id_agents_per_env = [env.action_space.spaces.keys() for env in envs]
gfollow = GapFollower()

max_steps = 500
for _ in range(1):
    # initialize episode datastructures
    both_done, both_episode, both_rewards = [], [], []
    remaining_steps = max_steps
    for env in envs:
        both_done.append(False)
        both_episode.append([env.reset()])
        both_rewards.append([])
    # run episode
    while not any(both_done) and remaining_steps>0:
        remaining_steps = remaining_steps - 1
        for i, (env, n_agt) in enumerate(zip(envs, id_agents_per_env)):
            multi_action = dict()
            for agent_id in n_agt:
                act = gfollow.action(both_episode[i][-1][agent_id])     # choose next action from the last observation
                act = {'motor': (act[0], act[1]), 'steering': act[2]}
                multi_action[agent_id] = act

            obs, rewards, dones, states = env.step(multi_action)
            both_done[i] = any(dones.values())
            both_episode[i].append(obs)
            both_rewards[i].append(rewards)

for env in envs:
    env.close()

train_rewards_a = [reward['A'] for reward in both_rewards[0]]
test_rewards_a = [reward['A'] for reward in both_rewards[1]]
plt.plot(np.arange(len(both_rewards[0])), train_rewards_a, label="Train Reward")
plt.plot(np.arange(len(both_rewards[1])), test_rewards_a, label="Test Reward")
plt.show()
