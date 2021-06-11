import unittest

import gym

from racecar_gym import MultiAgentScenario, MultiAgentRaceEnv
from racecar_gym.wrappers import SingleAgentWrapper

class SingleAgentAdapterTest(unittest.TestCase):

    def setUp(self) -> None:
        scenario = MultiAgentScenario.from_spec(path='./wrappers/scenarios/austria.yml')
        env = MultiAgentRaceEnv(scenario)
        self.env = SingleAgentWrapper(env=env, agent_id='A')

    def test_single_agent_adapter_spaces(self):
        self.assertTrue(isinstance(self.env.observation_space, gym.spaces.Dict))
        self.assertTrue(isinstance(self.env.action_space, gym.spaces.Dict))
        self.check_actions(actions=self.env.action_space.sample())
        self.check_observations(observations=self.env.observation_space.sample())

    def test_reset(self):
        obs = self.env.reset(mode='grid')
        self.check_observations(observations=obs)

    def test_step(self):
        _ = self.env.reset()
        obs, reward, done, info = self.env.step(self.env.action_space.sample())
        self.check_observations(observations=obs)
        self.assertTrue(isinstance(reward, float))
        self.assertTrue(isinstance(done, bool))

    def test_render(self):
        _ = self.env.reset()
        image = self.env.render(mode='follow', width=32, height=64)
        self.assertEqual((64,32,3), image.shape)

    def check_actions(self, actions):
        for key in ['steering', 'motor']:
            self.assertIn(key, actions)

    def check_observations(self, observations):
        for key in ['lidar', 'pose', 'velocity', 'acceleration']:
            self.assertIn(key, observations)

if __name__ == '__main__':
    unittest.main()