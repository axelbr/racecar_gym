from typing import List, Tuple, Dict

import gym
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from .multi_agent_race import MultiAgentScenario, MultiAgentRaceEnv


class ForkedMultiAgentRaceEnv(gym.Env):

    def __init__(self, scenario: MultiAgentScenario, debug=False):
        self._env = None
        self._debug = debug

        parent_conn, child_conn = Pipe()
        self._env_connection = parent_conn
        env_process = Process(target=self._run_env, args=(scenario, child_conn))
        self._env = env_process
        self._env.start()

        spaces = self._env_connection.recv()
        obs_space, action_space = spaces
        self.observation_space = obs_space
        self.action_space = action_space

    def _run_env(self, scenario: MultiAgentScenario, connection: Connection):
        env = MultiAgentRaceEnv(scenario=scenario)
        _ = env.reset()
        if self._debug:
            print(f'env-{id}: Send observation and action space.')
        connection.send((env.observation_space, env.action_space))
        terminate = False
        while not terminate:
            if self._debug:
                print(f'env-{id}: Wait for action.')
            action = connection.recv()
            if self._debug:
                print(f'env-{id}: Received action: {action} Taking step.')
            step = env.step(action)
            if self._debug:
                print(f'env-{id}: Send step return.')
            connection.send(step)
            if self._debug:
                print(f'env-{id}: Should reset? Wait for signal.')
            do_reset = connection.recv()
            if self._debug:
                print(f'env-{id}: {"Do not" if not do_reset else "Do"} reset.')
            if do_reset == True:
                obs = env.reset()
                if self._debug:
                    print(f'env-{id}: Did reset. Send reset result.')
                connection.send(obs)
            if self._debug:
                print(f'env-{id}: Should terminate? Wait for signal.')
            terminate = connection.recv()
        if self._debug:
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
        conn.send(self.action_space.sample())
        _ = conn.recv()
        conn.send(True)
        obs = conn.recv()
        conn.send(False)
        return obs

    def close(self):
        conn = self._env_connection
        conn.send(self.action_space.sample())
        conn.send(False)
        conn.send(True)
        conn.close()