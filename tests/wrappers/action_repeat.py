import unittest
import gym
from racecar_gym import MultiAgentScenario, SingleAgentScenario, MultiAgentRaceEnv, SingleAgentRaceEnv, \
    VectorizedSingleAgentRaceEnv
from racecar_gym.envs.gym_api import wrappers


class SingleActionRepeatStub(gym.Wrapper):

    def __init__(self, env, step_reward=1.0):
        super().__init__(env)
        self.calls = 0
        self._step_reward = step_reward

    def step(self, action):
        self.calls += 1
        obs, reward, done, info = self.env.step(action)
        return obs, self._step_reward, done, info

class MultiActionRepeatStub(gym.Wrapper):

    def __init__(self, env, step_reward=1.0):
        super().__init__(env)
        self.calls = 0
        self._step_reward = step_reward

    def step(self, action):
        self.calls += 1
        obs, reward, done, info = self.env.step(action)
        reward = dict((k, self._step_reward) for k in reward.keys())
        return obs, reward, done, info

class VectorizedSingleActionRepeatStub(gym.Wrapper):

    def __init__(self, env, step_reward=1.0):
        super().__init__(env)
        self.calls = 0
        self._step_reward = step_reward

    def step(self, action):
        self.calls += 1
        obs, reward, done, info = self.env.step(action)
        return obs, len(reward) * [self._step_reward], done, info


class ActionRepeatTest(unittest.TestCase):

    def setUp(self) -> None:
        self._scenario = './wrappers/scenarios/austria.yml'


    def test_single_agent_action_repeat(self):
        scenario = SingleAgentScenario.from_spec(path=self._scenario)
        env = SingleAgentRaceEnv(scenario)
        stub = SingleActionRepeatStub(env, step_reward=1.0)
        env = wrappers.SingleAgentActionRepeat(stub, steps=4)
        env.reset()
        obs, reward, done, info = env.step(env.action_space.sample())
        self.assertEqual(4, stub.calls)
        self.assertAlmostEqual(4.0, reward)

    def test_multi_agent_action_repeat(self):
        scenario = MultiAgentScenario.from_spec(path=self._scenario)
        env = MultiAgentRaceEnv(scenario)
        stub = MultiActionRepeatStub(env, step_reward=1.0)
        env = wrappers.MultiAgentActionRepeat(stub, steps=4)
        env.reset()
        obs, reward, done, info = env.step(env.action_space.sample())
        self.assertEqual(4, stub.calls)
        for agent_reward in reward.values():
            self.assertAlmostEqual(4.0, agent_reward)

    def test_vectorized_single_agent_action_repeat(self):
        scenario = SingleAgentScenario.from_spec(path=self._scenario)
        env = VectorizedSingleAgentRaceEnv(scenarios=[scenario, scenario])
        stub = VectorizedSingleActionRepeatStub(env, step_reward=1.0)
        env = wrappers.VectorizedSingleAgentActionRepeat(stub, steps=4)
        env.reset()
        obs, reward, done, info = env.step(env.action_space.sample())
        self.assertEqual(4, stub.calls)
        for i in range(2):
            self.assertAlmostEqual(4.0, reward[i])
        env.close()