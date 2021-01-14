from racecar_gym.envs import ChangingTrackMultiAgentRaceEnv, ChangingTrackSingleAgentRaceEnv
from racecar_gym import MultiAgentScenario, SingleAgentScenario
from agents.gap_follower import GapFollower

scenarios = [SingleAgentScenario.from_spec(f'scenarios/{track}.yml', rendering=True) for track in ['austria', 'gbr', 'barcelona']]
env = ChangingTrackSingleAgentRaceEnv(scenarios=scenarios, order='random')

for i in range(100):
    obs = env.reset()
    for _ in range(500):
        action = env.action_space.sample()
        _ = env.step(action)
env.close()