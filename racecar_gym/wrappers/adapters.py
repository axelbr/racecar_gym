import gym
from gym import Wrapper

from racecar_gym.envs import MultiAgentRaceEnv


class SingleAgentWrapper(Wrapper):

    def __init__(self, env: MultiAgentRaceEnv, agent_id: str):
        super().__init__(env)
        self._id = agent_id
        self._other_agents = set(env.observation_space.spaces.keys()) - {agent_id}
        self.observation_space = env.observation_space.spaces[agent_id]
        self.action_space = env.action_space.spaces[agent_id]

    def _build_spaces(self, base_env: gym.Env, horizon: int):
        new_action_space = dict()
        for agent in base_env.observation_space.spaces:
            new_action_space[agent] = gym.spaces.Dict()
            obs_space = base_env.observation_space.spaces[agent]
            for obs_key, space in obs_space.spaces.items():
                if isinstance(space, gym.spaces.Box):
                    low = np.repeat([space.low], horizon, axis=0)
                    high = np.repeat([space.high], horizon, axis=0)
                    new_obs_space[agent].spaces[obs_key] = gym.spaces.Box(low, high)
                else:
                    raise NotImplementedError('No other spaces are yet implemented for stacking!')

        self.observation_space = gym.spaces.Dict(new_obs_space)


    def step(self, action):
        actions = dict((agent, self.action_space.sample()) for agent in self._other_agents)
        actions[self._id] = action
        step = self.env.step(actions)
        return tuple(elem[self._id] for elem in step)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)[self._id]

    def render(self, mode='follow', **kwargs):
        return self.env.render(mode, **kwargs)