from typing import List, Tuple, Dict

import gym
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from racecar_gym import MultiAgentScenario, MultiAgentRaceEnv


class VectorizedMultiAgentRaceEnv(gym.Env):

    def __init__(self, scenarios: List[MultiAgentScenario]):
        self._env_connections = []
        self._envs = []
        for i, scenario in enumerate(scenarios):
            parent_conn, child_conn = Pipe()
            self._env_connections.append(parent_conn)
            env_process = Process(target=self._run_env, args=(scenario, child_conn, i))
            self._envs.append(env_process)
            env_process.start()

        spaces = [c.recv() for c in self._env_connections]
        obs_spaces, action_spaces = tuple(zip(*spaces))
        self.observation_space = gym.spaces.Tuple(obs_spaces)
        self.action_space = gym.spaces.Tuple(action_spaces)

    def _run_env(self, scenario: MultiAgentScenario, connection: Connection, id: int):
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

    def step(self, actions: Tuple[Dict]):

        for action, conn in zip(actions, self._env_connections):
            conn.send(action)

        observations, rewards, dones, states = [], [], [], []
        for conn in self._env_connections:
            obs, reward, done, state = conn.recv()
            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            states.append(state)

        for conn in self._env_connections:
            conn.send(False)
            conn.send(False)

        return observations, rewards, dones, states

    def reset(self):
        observations = []
        for i, conn in enumerate(self._env_connections):
            conn.send(self.action_space.spaces[i].sample())
            _ = conn.recv()
            conn.send(True)
            obs = conn.recv()
            observations.append(obs)
            conn.send(False)
        return observations

    def close(self):
        for i, conn in enumerate(self._env_connections):
            conn.send(self.action_space.spaces[i].sample())
            conn.send(False)
            conn.send(True)
            conn.close()