from typing import List, Tuple, Dict

import gym
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from racecar_gym import MultiAgentScenario, MultiAgentRaceEnv


class ForkedMultiAgentRaceEnv(gym.Env):

    def __init__(self, scenario: MultiAgentScenario):
        self._env_connections = []
        self._env = None

        parent_conn, child_conn = Pipe()
        self._env_connection = parent_conn
        env_process = Process(target=self._run_env, args=(scenario, child_conn))
        self._env = env_process
        self._env.start()

        spaces = [c.recv() for c in self._env_connections]
        obs_spaces, action_spaces = tuple(zip(*spaces))
        self.observation_space = gym.spaces.Tuple(obs_spaces)
        self.action_space = gym.spaces.Tuple(action_spaces)

    def _run_env(self, scenario: MultiAgentScenario, connection: Connection):
        env = MultiAgentRaceEnv(scenario=scenario)
        _ = env.reset()
        print(f'env-{id}: Send observation and action space.')
        connection.send((env.observation_space, env.action_space))
        terminate = False
        while not terminate:
            print(f'env-{id}: Wait for action.')
            action = connection.recv()
            print(f'env-{id}: Received action: {action} Taking step.')
            step = env.step(action)
            print(f'env-{id}: Send step return.')
            connection.send(step)
            print(f'env-{id}: Should reset? Wait for signal.')
            do_reset = connection.recv()
            print(f'env-{id}: {"Do not" if not do_reset else "Do"} reset.')
            if do_reset == True:
                obs = env.reset()
                print(f'env-{id}: Did reset. Send reset result.')
                connection.send(obs)
            print(f'env-{id}: Should terminate? Wait for signal.')
            terminate = connection.recv()
        print(f'env-{id}: Terminating.')

    def step(self, actions: Dict):
        conn = self._env_connection
        conn.send(actions)
        obs, reward, done, state = conn.recv()
        conn.send(False)
        conn.send(False)
        return obs, reward, done, state

    def reset(self):
        conn = self._env_connection
        conn.send(self.action_space.spaces[i].sample())
        _ = conn.recv()
        conn.send(True)
        obs = conn.recv()
        conn.send(False)
        return obs

    def close(self):
        conn = self._env_connection
        conn.send(self.action_space.spaces.sample())
        conn.send(False)
        conn.send(True)
        conn.close()