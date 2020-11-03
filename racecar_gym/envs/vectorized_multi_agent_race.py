from typing import List, Tuple, Dict

import gym
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from .scenarios import MultiAgentScenario
from .multi_agent_race import MultiAgentRaceEnv


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
        print('Run environment.')
        env = MultiAgentRaceEnv(scenario=scenario)
        _ = env.reset()
        connection.send((env.observation_space, env.action_space))
        terminate = False
        while not terminate:
            action = connection.recv()
            step = env.step(action)
            connection.send(step)
            do_reset = connection.recv()
            if do_reset == True:
                obs = env.reset()
                connection.send(obs)
            terminate = connection.recv()

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