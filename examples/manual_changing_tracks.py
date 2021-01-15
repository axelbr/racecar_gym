from racecar_gym.envs import ChangingTrackMultiAgentRaceEnv, ChangingTrackSingleAgentRaceEnv
from racecar_gym import MultiAgentScenario, SingleAgentScenario
from agents.gap_follower import GapFollower

scenarios = [MultiAgentScenario.from_spec(f'scenarios/{track}.yml', rendering=True) for track in ['austria', 'barcelona']]
env = ChangingTrackMultiAgentRaceEnv(scenarios=scenarios, order='manual')

for _ in range(4):
    env.set_next_env()
    for i in range(2):
        obs = env.reset()
        for _ in range(500):
            action = env.action_space.sample()
            _ = env.step(action)

env.close()