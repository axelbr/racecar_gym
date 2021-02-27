from agents.gap_follower import GapFollower
from time import sleep
from racecar_gym import MultiAgentScenario
from racecar_gym.envs import MultiAgentRaceEnv

scenario = MultiAgentScenario.from_spec("scenarios/custom.yml", rendering=True)
env = MultiAgentRaceEnv(scenario=scenario)

agent = GapFollower(fixed_speed=False)
done = False
obs = env.reset(mode='random')
t = 0
while not done:
    action = env.action_space.sample()
    action_gf = agent.action(obs['A'])
    action['A'] = {'motor': action_gf[0], 'steering': action_gf[1]}
    obs, rewards, dones, states = env.step(action)
    done = any(dones.values())
    #sleep(0.01)
    if t % 10 == 0:
        image = env.render(mode='follow', agent='A')
    t += 1


env.close()