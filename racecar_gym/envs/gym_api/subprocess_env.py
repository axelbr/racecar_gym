from multiprocessing.connection import Connection, Pipe
from multiprocessing.context import Process
from typing import Union, Any, Callable

from gymnasium import Env

class SubprocessEnv(Env):

    def __init__(self, factory: Callable[[], Env], blocking: bool = True):
        self._blocking = blocking
        self._parent_conn, child_conn = Pipe()
        self._process = Process(target=self._start, args=(factory, child_conn))
        self._process.start()
        self.observation_space, self.action_space = self._parent_conn.recv()

    def _start(self, factory: Callable[[], Env], connection: Connection):
        env = factory()
        _ = env.reset()
        connection.send((env.observation_space, env.action_space))
        terminate = False
        while not terminate:
            command, kwargs = connection.recv()
            if command == 'render':
                rendering = env.render(**kwargs)
                connection.send(rendering)
            elif command == 'step':
                step = env.step(**kwargs)
                connection.send(step)
            elif command == 'reset':
                obs = env.reset(**kwargs)
                connection.send(obs)
            elif command == 'close':
                terminate = True
                connection.close()

    def step(self, action):
        self._parent_conn.send(('step', dict(action=action)))
        return self._return()

    def reset(self, **kwargs):
        self._parent_conn.send(('reset', kwargs))
        return self._return()

    def render(self, **kwargs):
        self._parent_conn.send(('render', {**kwargs}))
        return self._return()

    def close(self):
        self._parent_conn.send(('close', False))
        self._parent_conn.close()

    def _return(self) -> Any:
        if self._blocking:
            return self._parent_conn.recv()
        else:
            return lambda: self._parent_conn.recv()

