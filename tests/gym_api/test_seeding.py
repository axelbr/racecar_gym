import random
from unittest.mock import MagicMock, call, patch

import gymnasium
import numpy as np

from racecar_gym.bullet.world import World
from racecar_gym.envs.gym_api.multi_agent_race import MultiAgentRaceEnv
from racecar_gym.envs.gym_api.changing_track_race_env import ChangingTrackRaceEnv
from racecar_gym.envs.gym_api.single_agent_race import SingleAgentRaceEnv


def test_single_agent_reset_seeds_world_before_sampling_starting_position():
    env = object.__new__(SingleAgentRaceEnv)
    gymnasium.Env.__init__(env)
    env._initialized = False
    env._scenario = MagicMock()
    env._scenario.agent.id = "A"
    env._scenario.agent.reset.return_value = {}
    env._scenario.world.state.return_value = {"A": {"time": 0.01}}

    manager = MagicMock()
    manager.attach_mock(env._scenario.world.seed, "seed")
    manager.attach_mock(env._scenario.world.get_starting_position, "starting_position")

    env.reset(seed=42, options={"mode": "random"})

    assert manager.mock_calls[-2:] == [
        call.seed(42),
        call.starting_position(env._scenario.agent, "random"),
    ]


def test_multi_agent_reset_seeds_world_before_sampling_starting_positions():
    env = object.__new__(MultiAgentRaceEnv)
    gymnasium.Env.__init__(env)
    env._initialized = False
    agent = MagicMock()
    agent.id = "A"
    agent.reset.return_value = {}
    env._scenario = MagicMock()
    env._scenario.agents = {"A": agent}
    env._scenario.world.state.return_value = {"A": {"time": 0.01}}

    manager = MagicMock()
    manager.attach_mock(env._scenario.world.seed, "seed")
    manager.attach_mock(env._scenario.world.get_starting_position, "starting_position")

    env.reset(seed=42, options={"mode": "random"})

    assert manager.mock_calls[-2:] == [
        call.seed(42),
        call.starting_position(agent=agent, mode="random"),
    ]


def _world_without_assets():
    world = object.__new__(World)
    world._agents = [MagicMock()]
    return world


def test_world_seed_owns_random_state_without_mutating_process_generators():
    np.random.seed(7)
    expected_numpy_values = np.random.random(3)
    np.random.seed(7)
    random.seed(7)
    expected_python_values = [random.random() for _ in range(3)]
    random.seed(7)

    world = _world_without_assets()
    world.seed(42)

    np.testing.assert_array_equal(np.random.random(3), expected_numpy_values)
    assert [random.random() for _ in range(3)] == expected_python_values


def test_seeded_world_generators_are_isolated_when_interleaved():
    world = _world_without_assets()
    other_world = _world_without_assets()
    reference_world = _world_without_assets()
    world.seed(42)
    other_world.seed(99)
    reference_world.seed(42)

    values = [world._np_random.random(), world._random.random()]
    other_world._np_random.random()
    other_world._random.random()
    values.extend([world._np_random.random(), world._random.random()])

    expected = [
        reference_world._np_random.random(),
        reference_world._random.random(),
        reference_world._np_random.random(),
        reference_world._random.random(),
    ]
    np.testing.assert_array_equal(values, expected)
    world._agents[0].seed.assert_called_once_with(world._np_random)


@patch("racecar_gym.envs.gym_api.changing_track_race_env.SubprocessEnv")
def test_random_track_order_is_repeatable_for_fresh_environments(subprocess_env):
    subprocess_env.return_value.action_space = gymnasium.spaces.Discrete(1)
    subprocess_env.return_value.observation_space = gymnasium.spaces.Discrete(1)
    subprocess_env.return_value.reset.return_value = ({}, {})
    factories = [MagicMock(), MagicMock(), MagicMock()]
    first = ChangingTrackRaceEnv(factories, order="random")
    second = ChangingTrackRaceEnv(factories, order="random")

    first.reset(seed=42)
    second.reset(seed=42)

    assert first._current_track_index == second._current_track_index
