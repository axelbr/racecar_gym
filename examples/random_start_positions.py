from time import sleep

from racecar_gym import MultiAgentScenario
from racecar_gym.envs.multi_race_car_env import MultiAgentRaceCarEnv

scenario = MultiAgentScenario.from_spec(
    path='random_starts.yml',
    rendering=True
)
env = MultiAgentRaceCarEnv(scenario=scenario)

print(env.observation_space)
print(env.action_space)

done = False
obs = env.reset()

for _ in range(10):
    env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        obs, rewards, dones, states = env.step(action)
        done = any(dones.values())
        sleep(0.01)
env.close()
