from pathlib import Path

from pettingzoo import ParallelEnv

from racecar_gym.envs import pettingzoo_api


def test_raw_env_factory_returns_parallel_environment():
    scenario = Path(__file__).with_name("austria.yml")

    env = pettingzoo_api.raw_env(
        scenario=str(scenario),
        render_mode="rgb_array_follow",
    )

    try:
        assert isinstance(env, ParallelEnv)
    finally:
        env.close()
