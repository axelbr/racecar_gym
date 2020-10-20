from racecar_gym import MultiAgentScenario
from racecar_gym.envs.vectorized_multi_agent_race import VectorizedMultiAgentRaceEnv

scenarios = [MultiAgentScenario.from_spec('custom.yml', rendering=True) for _ in range(2)]
env = VectorizedMultiAgentRaceEnv(scenarios=scenarios)
for i in range(3):
    done = False
    _ = env.reset()
    while not done:
        action = env.action_space.sample()
        obs, rewards, dones, states = env.step(action)
        done = any([any(e.values()) for e in dones])
env.close()