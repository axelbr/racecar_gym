import gymnasium
import numpy as np

import racecar_gym.envs.gym_api  # noqa: F401


def test_first_seeded_reset_matches_the_next_seeded_reset():
    env = gymnasium.make(
        "SingleAgentAustria-v0",
        render_mode="rgb_array_follow",
        disable_env_checker=True,
    )
    try:
        first_observation, first_state = env.reset(seed=42, options={"mode": "grid"})
        second_observation, second_state = env.reset(seed=42, options={"mode": "grid"})
    finally:
        env.close()

    for key, expected in first_observation.items():
        np.testing.assert_array_equal(second_observation[key], expected)
    for key in ("pose", "velocity", "acceleration"):
        np.testing.assert_array_equal(second_state[key], first_state[key])
