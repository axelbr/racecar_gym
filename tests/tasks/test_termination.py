from racecar_gym.tasks.progress_based import MaximizeProgressTask


def _state(*, lap: int = 1, time: float = 0.0):
    return {
        "agent": {
            "lap": lap,
            "progress": 0.5,
            "time": time,
            "wall_collision": False,
            "opponent_collisions": [],
        }
    }


def _task():
    return MaximizeProgressTask(
        laps=1,
        time_limit=60.0,
        terminate_on_collision=False,
    )


def test_time_limit_truncates_instead_of_terminating():
    task = _task()
    state = _state(time=61.0)

    assert task.done("agent", state) is False
    assert task.truncated("agent", state) is True


def test_lap_completion_terminates_without_truncating():
    task = _task()
    state = _state(lap=2)

    assert task.done("agent", state) is True
    assert task.truncated("agent", state) is False
