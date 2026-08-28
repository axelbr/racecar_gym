import numpy as np
import pytest
from unittest.mock import patch

from racecar_gym.bullet.vehicle import RaceCar


def test_random_vehicle_color_uses_seeded_vehicle_generator():
    first = RaceCar([], [], RaceCar.Config("racecar.urdf", "random"))
    second = RaceCar([], [], RaceCar.Config("racecar.urdf", "random"))
    first.seed(np.random.default_rng(42))
    second.seed(np.random.default_rng(42))

    np.testing.assert_array_equal(first._color(), second._color())


def test_unknown_vehicle_color_fails_clearly():
    vehicle = RaceCar([], [], RaceCar.Config("racecar.urdf", "invisible"))

    with pytest.raises(ValueError, match="Unsupported vehicle color: invisible"):
        vehicle._color()


@patch("racecar_gym.bullet.vehicle.pybullet")
def test_random_color_consumes_same_seed_draws_on_every_reset(pybullet):
    pybullet.getQuaternionFromEuler.return_value = "quaternion"
    pybullet.getNumJoints.return_value = 0
    vehicle = RaceCar([], [], RaceCar.Config("racecar.urdf", "random"))
    vehicle._id = 7
    pose = ((1.0, 2.0, 0.05), (0.0, 0.0, 0.5))

    vehicle.seed(np.random.default_rng(42))
    vehicle.reset(pose)
    first_color = pybullet.changeVisualShape.call_args.kwargs["rgbaColor"]
    vehicle.seed(np.random.default_rng(42))
    vehicle.reset(pose)
    second_color = pybullet.changeVisualShape.call_args.kwargs["rgbaColor"]

    np.testing.assert_array_equal(first_color, second_color)
