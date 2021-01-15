from time import sleep
from racecar_gym import MultiAgentScenario
from racecar_gym.envs.multi_agent_race import MultiAgentRaceEnv

scenario = MultiAgentScenario.from_spec(
    path='scenarios/custom.yml',
    rendering=True
)

env = MultiAgentRaceEnv(scenario=scenario)

print(env.observation_space)
print(env.action_space)

done = False
obs = env.reset(mode='random_ball')

while not done:
    action = env.action_space.sample()
    obs, rewards, dones, states = env.step(action)
    done = any(dones.values())
    sleep(0.01)

env.close()
