import gymnasium
import numpy as np

import racecar_gym.envs.gym_api  # noqa: F401


def _reset_snapshot(env):
    env.unwrapped.scenario.world.seed(42)
    observation, info = env.reset(options={"mode": "grid"})
    return (
        {key: np.array(value, copy=True) for key, value in observation.items()},
        {
            key: np.array(info[key], copy=True)
            for key in ("pose", "velocity", "acceleration")
        },
    )


def test_episode_reset_discards_pybullet_and_sensor_state():
    env = gymnasium.make(
        "SingleAgentAustria-v0",
        render_mode="rgb_array_follow",
        disable_env_checker=True,
    )
    try:
        env.reset(options={"mode": "grid"})  # Load the PyBullet bodies once.

        expected_observation, expected_state = _reset_snapshot(env)
        action = {
            "motor": np.array([1.0], dtype=np.float32),
            "steering": np.array([0.3], dtype=np.float32),
        }
        for _ in range(200):
            env.step(action)

        actual_observation, actual_state = _reset_snapshot(env)
    finally:
        env.close()

    for key, expected in expected_observation.items():
        np.testing.assert_array_equal(actual_observation[key], expected)
    for key, expected in expected_state.items():
        np.testing.assert_array_equal(actual_state[key], expected)


def test_multi_agent_reset_observes_completed_starting_layout():
    env = gymnasium.make(
        "MultiAgentAustria-v0",
        render_mode="rgb_array_follow",
        disable_env_checker=True,
    )
    try:
        env.reset(seed=0, options={"mode": "grid"})
        env.unwrapped.scenario.world.seed(42)
        expected_observations, expected_state = env.reset(options={"mode": "grid"})
        expected_observations = {
            agent: {key: np.array(value, copy=True) for key, value in observation.items()}
            for agent, observation in expected_observations.items()
        }
        expected_state = {
            agent: np.array(state["pose"], copy=True)
            for agent, state in expected_state.items()
            if agent != "maps"
        }
        action = {
            agent: {
                "motor": np.array([1.0], dtype=np.float32),
                "steering": np.array([0.3], dtype=np.float32),
            }
            for agent in env.action_space.spaces
        }
        for _ in range(200):
            env.step(action)

        env.unwrapped.scenario.world.seed(42)
        actual_observations, actual_state = env.reset(options={"mode": "grid"})
    finally:
        env.close()

    for agent, expected in expected_observations.items():
        for key, expected_value in expected.items():
            np.testing.assert_array_equal(actual_observations[agent][key], expected_value)
        np.testing.assert_array_equal(actual_state[agent]["pose"], expected_state[agent])
