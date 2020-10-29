from time import sleep
from racecar_gym import MultiAgentScenario
from racecar_gym.envs.multi_agent_race import MultiAgentRaceEnv

scenario = MultiAgentScenario.from_spec(
                 path='scenarios/berlin.yml',
                 rendering=True
             )
env = MultiAgentRaceEnv(scenario=scenario)

print(env.observation_space)
print(env.action_space)

for i in range(10):
    done = False
    obs = env.reset()
    while not done:
        action = env.action_space.sample()
        obs, rewards, dones, states = env.step(action)
        done = any(dones.values())
        sleep(0.01)

env.close()
