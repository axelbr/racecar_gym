from typing import List, Tuple, Dict

import gym
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from .scenarios import SingleAgentScenario
from .single_agent_race import SingleAgentRaceEnv

class VectorizedSingleAgentRaceEnv(gym.Env):

    metadata = {'render.modes': ['follow', 'birds_eye']}

    def __init__(self, scenarios: List[SingleAgentScenario]):
        self._env_connections = []
        self._envs = []
        for i, scenario in enumerate(scenarios):
            parent_conn, child_conn = Pipe()
            self._env_connections.append(parent_conn)
            env_process = Process(target=self._run_env, args=(scenario, child_conn))
            self._envs.append(env_process)
            env_process.start()

        spaces = [c.recv() for c in self._env_connections]
        obs_spaces, action_spaces = tuple(zip(*spaces))
        self.observation_space = gym.spaces.Tuple(obs_spaces)
        self.action_space = gym.spaces.Tuple(action_spaces)

    def _run_env(self, scenario: SingleAgentScenario, connection: Connection):
        print('Run environment.')
        env = SingleAgentRaceEnv(scenario=scenario)
        _ = env.reset()
        connection.send((env.observation_space, env.action_space))
        terminate = False
        while not terminate:
            command = connection.recv()
            if command == 'render':
                mode = connection.recv()
                rendering = env.render(mode)
                connection.send(rendering)
            elif command == 'step':
                action = connection.recv()
                step = env.step(action)
                connection.send(step)
            elif command == 'reset':
                obs = env.reset()
                connection.send(obs)
            elif command == 'close':
                terminate = True

    def step(self, actions: Tuple[Dict]):

        for action, conn in zip(actions, self._env_connections):
            conn.send('step')
            conn.send(action)

        observations, rewards, dones, states = [], [], [], []
        for conn in self._env_connections:
            obs, reward, done, state = conn.recv()
            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            states.append(state)

        return observations, rewards, dones, states

    def reset(self):
        observations = []
        for i, conn in self._env_connections:
            conn.send('reset')
            obs = conn.recv()
            observations.append(obs)
        return observations

    def close(self):
        for conn in self._env_connections:
            conn.send('close')
            conn.close()

    def render(self, mode='follow'):
        renderings = []
        for conn in self._env_connections:
            conn.send('render')
            conn.send(mode)

        for conn in self._env_connections:
            rendering = conn.recv()
            renderings.append(rendering)

        return renderings

