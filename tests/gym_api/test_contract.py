from unittest.mock import MagicMock

import gymnasium
import numpy as np

from racecar_gym.envs.gym_api.single_agent_race import SingleAgentRaceEnv
from racecar_gym.envs.gym_api.multi_agent_race import MultiAgentRaceEnv
from racecar_gym.envs.gym_api.wrappers.action_repeat import SingleAgentActionRepeat
from racecar_gym.envs.gym_api.wrappers.action_repeat import VectorizedSingleAgentActionRepeat
from racecar_gym.envs.pettingzoo_api.racecarenv import _MultiAgentRaceEnv


class _TruncatingEnv(gymnasium.Env):
    action_space = gymnasium.spaces.Discrete(1)
    observation_space = gymnasium.spaces.Discrete(1)

    def __init__(self):
        self.calls = 0

    def step(self, action):
        self.calls += 1
        return 0, 1.0, False, True, {}


def test_time_observation_space_accepts_elapsed_seconds_above_one():
    env = SingleAgentRaceEnv.__new__(SingleAgentRaceEnv)
    env._scenario = MagicMock()
    env._scenario.agent.observation_space = gymnasium.spaces.Dict({})
    env._scenario.world.space.return_value = gymnasium.spaces.Dict({
        "time": gymnasium.spaces.Box(0.0, np.inf, shape=(), dtype=np.float32),
    })

    time_space = env.observation_space["time"]

    assert time_space.contains(np.array(2.0, dtype=np.float32))


def test_single_agent_step_forwards_termination_and_truncation_separately():
    env = SingleAgentRaceEnv.__new__(SingleAgentRaceEnv)
    env._initialized = True
    env._scenario = MagicMock()
    env._scenario.agent.id = "agent"
    env._scenario.agent.step.return_value = ({}, {})
    env._scenario.agent.done.return_value = False
    env._scenario.agent.truncated.return_value = True
    env._scenario.agent.reward.return_value = 1.0
    env._scenario.world.state.return_value = {"agent": {"time": 61.0}}

    _, _, terminated, truncated, _ = env.step(action={})

    assert terminated is False
    assert truncated is True


def test_multi_agent_step_returns_truncation_for_each_agent():
    env = MultiAgentRaceEnv.__new__(MultiAgentRaceEnv)
    env._initialized = True
    agent = MagicMock()
    agent.step.return_value = ({}, {})
    agent.done.return_value = False
    agent.truncated.return_value = True
    agent.reward.return_value = 1.0
    env._scenario = MagicMock()
    env._scenario.agents = {"agent": agent}
    env._scenario.world.state.return_value = {"agent": {"time": 61.0}}

    _, _, terminations, truncations, _ = env.step({"agent": {}})

    assert terminations == {"agent": False}
    assert truncations == {"agent": True}


def test_action_repeat_stops_on_truncation():
    inner_env = _TruncatingEnv()
    env = SingleAgentActionRepeat(inner_env, steps=4)

    _, reward, terminated, truncated, _ = env.step(0)

    assert inner_env.calls == 1
    assert reward == 1.0
    assert terminated is False
    assert truncated is True


class _MixedCompletionVectorEnv(gymnasium.Env):
    action_space = gymnasium.spaces.Tuple((gymnasium.spaces.Discrete(1),) * 2)
    observation_space = gymnasium.spaces.Tuple((gymnasium.spaces.Discrete(1),) * 2)

    def __init__(self):
        self.calls = 0

    def step(self, action):
        self.calls += 1
        return (
            [0, 0],
            np.array([1.0, 1.0]),
            np.array([True, False]),
            np.array([False, True]),
            [{}, {}],
        )


def test_vectorized_action_repeat_stops_when_all_lanes_finish_for_mixed_reasons():
    inner_env = _MixedCompletionVectorEnv()
    env = VectorizedSingleAgentActionRepeat(inner_env, steps=4)

    _, reward, terminated, truncated, _ = env.step((0, 0))

    assert inner_env.calls == 1
    np.testing.assert_array_equal(reward, np.array([1.0, 1.0]))
    np.testing.assert_array_equal(terminated, np.array([True, False]))
    np.testing.assert_array_equal(truncated, np.array([False, True]))


def test_pettingzoo_removes_only_the_agent_that_finishes():
    env = _MultiAgentRaceEnv.__new__(_MultiAgentRaceEnv)
    env.possible_agents = ["A", "B"]
    env.agents = env.possible_agents[:]
    env._env = MagicMock()
    env._env.step_agents.side_effect = [
        (
            {"A": {}, "B": {}},
            {"A": 0.0, "B": 0.0},
            {"A": False, "B": False},
            {"A": True, "B": False},
            {"A": {}, "B": {}},
        ),
        (
            {"B": {}},
            {"B": 0.0},
            {"B": False},
            {"B": False},
            {"B": {}},
        ),
    ]

    _, _, terminations, truncations, _ = env.step({"A": {}, "B": {}})

    assert terminations == {"A": False, "B": False}
    assert truncations == {"A": True, "B": False}
    assert env.agents == ["B"]

    env.step({"B": {}})

    env._env.step_agents.assert_called_with({"B": {}}, ["B"])


def test_multi_agent_environment_stops_finished_agents_and_continues_live_agents():
    env = MultiAgentRaceEnv.__new__(MultiAgentRaceEnv)
    env._initialized = True
    first = MagicMock()
    second = MagicMock()
    first.step.return_value = ({}, {})
    second.step.return_value = ({}, {})
    first.done.return_value = False
    second.done.return_value = False
    first.truncated.return_value = True
    second.truncated.return_value = False
    first.reward.return_value = 0.0
    second.reward.return_value = 0.0
    env._scenario = MagicMock()
    env._scenario.agents = {"A": first, "B": second}
    env._scenario.world.state.return_value = {
        "A": {"time": 1.0},
        "B": {"time": 1.0},
    }

    env.step_agents({"A": {}, "B": {}}, ["A", "B"])
    first.truncated.return_value = False
    env.step_agents({"B": {}}, ["B"])

    assert first.step.call_count == 1
    assert second.step.call_count == 2
    first.stop.assert_called_once_with()


def test_multi_agent_gym_step_keeps_declared_fixed_spaces_after_partial_completion():
    env = MultiAgentRaceEnv.__new__(MultiAgentRaceEnv)
    env._initialized = True
    agents = {}
    for agent_id in ("A", "B"):
        agent = MagicMock()
        agent.step.return_value = ({"sensor": np.array([0.5], dtype=np.float32)}, {})
        agent.done.return_value = False
        agent.truncated.return_value = agent_id == "A"
        agent.reward.return_value = 0.0
        agent.action_space = gymnasium.spaces.Box(-1.0, 1.0, (1,), np.float32)
        agent.observation_space = gymnasium.spaces.Dict({
            "sensor": gymnasium.spaces.Box(0.0, 1.0, (1,), np.float32)
        })
        agents[agent_id] = agent
    env._scenario = MagicMock()
    env._scenario.agents = agents
    env._scenario.world.space.return_value = {
        "time": gymnasium.spaces.Box(-np.inf, np.inf, (), np.float32)
    }
    env._scenario.world.state.return_value = {
        "A": {"time": 1.0}, "B": {"time": 1.0}
    }
    env.action_space = gymnasium.spaces.Dict({
        agent_id: agent.action_space for agent_id, agent in agents.items()
    })
    actions = {
        "A": np.array([0.0], dtype=np.float32),
        "B": np.array([0.0], dtype=np.float32),
    }

    observations, _, _, _, _ = env.step(actions)

    assert env.action_space.contains(actions)
    assert env.observation_space.contains(observations)
    assert set(observations) == set(env._scenario.agents)
