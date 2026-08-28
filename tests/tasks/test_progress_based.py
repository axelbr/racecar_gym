import pytest

from racecar_gym.tasks.progress_based import MaximizeProgressTask


def _state(progress: float):
    return {
        "agent": {
            "lap": 1,
            "progress": progress,
            "time": 0.0,
            "wall_collision": False,
            "opponent_collisions": [],
        }
    }


@pytest.fixture
def task():
    return MaximizeProgressTask(
        laps=1,
        time_limit=60.0,
        terminate_on_collision=False,
        progress_reward=100.0,
    )


@pytest.mark.parametrize(
    ("initial_progress", "next_progress", "expected_reward"),
    [
        (0.20, 0.30, 10.0),
        (0.30, 0.20, -10.0),
        (0.99, 0.01, 2.0),
        (0.01, 0.99, -2.0),
    ],
)
def test_progress_reward_preserves_direction_across_start_line(
    task,
    initial_progress,
    next_progress,
    expected_reward,
):
    assert task.reward("agent", _state(initial_progress), action={}) == 0.0

    reward = task.reward("agent", _state(next_progress), action={})

    assert reward == pytest.approx(expected_reward)
