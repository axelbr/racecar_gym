from agents.gap_follower import GapFollower

from time import sleep
from racecar_gym import MultiAgentScenario, MultiAgentRaceEnv

from racecar_gym.tasks import Task, register_task
from racecar_gym.tasks.progress_based import MaximizeContinuousProgressTask

register_task(name='maximize_cont_progress', task=MaximizeContinuousProgressTask)

scenario = MultiAgentScenario.from_spec("custom.yml", rendering=True)
env: MultiAgentRaceEnv = MultiAgentRaceEnv(scenario=scenario)
agent = GapFollower()

done = False
obs = env.reset(mode='grid')
t = 0
while not done:
    action = env.action_space.sample()
    action_gf = agent.action(obs['A'])
    action['A'] = {'motor': action_gf[:2], 'steering': action_gf[-1]}
    obs, rewards, dones, states = env.step(action)
    done = any(dones.values())
    sleep(0.01)
    if t % 10 == 0:
        image = env.render(mode='follow', agent='A')
    t += 1


env.close()