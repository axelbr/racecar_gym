from unittest.mock import patch

import pytest

from racecar_gym.bullet.actuators import Motor, Speed, SteeringWheel


@pytest.mark.parametrize(
    ("actuator", "control_mode", "expected"),
    [
        (Motor("motor", Motor.Config(1.0, 10.0, 5.0)), "VELOCITY_CONTROL", {"targetVelocity": 0.0, "force": 0.0}),
        (Speed("speed", Speed.Config(1.0, 10.0, 5.0)), "VELOCITY_CONTROL", {"targetVelocity": 0.0, "force": 5.0}),
        (SteeringWheel("steering", SteeringWheel.Config(1.0, 0.5)), "POSITION_CONTROL", {"targetPosition": 0.0}),
    ],
)
@patch("racecar_gym.bullet.actuators.pybullet")
def test_stop_uses_physical_neutral_command(pybullet, actuator, control_mode, expected):
    actuator.reset(body_id=7, joint_indices=[2])
    setattr(pybullet, control_mode, control_mode)

    actuator.stop()

    pybullet.setJointMotorControl2.assert_called_once_with(7, 2, control_mode, **expected)
