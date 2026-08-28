from pathlib import Path

from pettingzoo.test import parallel_api_test

from racecar_gym.envs import pettingzoo_api


def test_parallel_api_contract():
    scenario = Path(__file__).with_name("austria.yml")
    env = pettingzoo_api.raw_env(
        scenario=str(scenario),
        render_mode="rgb_array_follow",
    )

    parallel_api_test(env, num_cycles=100)
