from time import sleep

from agents.gap_follower import GapFollower
from racecar_gym import MultiAgentScenario
from racecar_gym.envs.multi_agent_race import MultiAgentRaceEnv

scenario = MultiAgentScenario.from_spec(
    path='random_starts.yml',
    rendering=True
)
env = MultiAgentRaceEnv(scenario=scenario)

print(env.observation_space)
print(env.action_space)

done = False
obs = env.reset()
agent = GapFollower()
for _ in range(10):
    obs = env.reset()
    done = False
    while not done:
        action = agent.action(obs['A'])
        action = {'A': {
            'motor': (action[0], action[1]),
            'steering': action[2]
        }}
        obs, rewards, dones, states = env.step(action)
        print(rewards)
        done = any(dones.values())
        sleep(0.01)
env.close()
