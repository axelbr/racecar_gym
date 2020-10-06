from functools import partial
from time import time, sleep

import gym
from pandas import np
from stable_baselines.common.callbacks import EvalCallback

import racecar_gym
from agents.gap_follower import GapFollower
from stable_baselines.common.policies import MlpPolicy, FeedForwardPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, SAC

class SingleWrapper(gym.Env):

    def __init__(self, env):
        self.env = env
        self.action_space = gym.spaces.Box(
            np.append([2, 0.5], env.action_space['A']['steering'].low),
            np.append(env.action_space['A']['motor'].high, env.action_space['A']['steering'].high))
        self.observation_space = env.observation_space['A']['lidar']

    def step(self, action):
        action = {'motor': (action[0], action[1]), 'steering': action[2]}
        obs, reward, done, info = self.env.step({'A': action})
        return obs['A']['lidar'], reward['A'], done['A'], info['A']

    def reset(self):
        obs = self.env.reset()
        return obs['A']['lidar']

    def render(self, mode='human'):
        pass

class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=[256, 256, 128, 64],
                                                          vf=[256, 256, 128, 64])],
                                           feature_extraction="mlp")

env = gym.make('MultiAgentTrack1_Gui-v0')
monitor_env = env  # wrappers.Monitor(env, directory='../recordings', force=True, video_callable=lambda episode_id: True)
#env.render()
observation = monitor_env.reset()
agent = GapFollower()
done = False

print(env.observation_space)
print(env.action_space)

i = 0
start = time()
rewards = dict([(id, 0) for id in observation.keys()])
images = []



# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])
env = SingleWrapper(env)
eval_callback = EvalCallback(env, n_eval_episodes=5, eval_freq=10000, verbose=1)
model = PPO2(CustomPolicy, env, verbose=1, tensorboard_log="./logs/", n_steps=6000)
model.load('logs/model')
model.learn(total_timesteps=1_000_000, tb_log_name="first_run", callback=eval_callback)
model.save('logs/model')

while not done:
    actions = {}
    for id, obs in observation.items():
        action = agent.action(obs)
        actions[id] = {'motor': (action[0], action[1]), 'steering': action[2]}
    observation, reward, dones, info = monitor_env.step(actions)
    # images.append(observation[1]['rgb_camera'])
    for id, reward in reward.items():
        rewards[id] += reward

    done = any(dones.values())
    print(rewards)
    i += 1
    #sleep(0.01)


#imageio.mimsave('movie.gif', images)
end = time()
print('wall time: ' + str((end-start)))
print('sim time: ' + str(i/100))
print('RTF: ', (i/100) / (end-start))
monitor_env.close()
env.close()
