from unittest.mock import call, patch

from racecar_gym.bullet.vehicle import RaceCar


@patch("racecar_gym.bullet.vehicle.pybullet")
def test_existing_vehicle_reset_clears_dynamic_state(pybullet):
    pybullet.getQuaternionFromEuler.return_value = "quaternion"
    pybullet.getNumJoints.return_value = 2
    vehicle = RaceCar(
        sensors=[],
        actuators=[],
        config=RaceCar.Config(urdf_file="racecar.urdf", color=(0.0, 0.0, 1.0, 1.0)),
    )
    vehicle._id = 7

    vehicle.reset(((1.0, 2.0, 0.05), (0.0, 0.0, 0.5)))

    pybullet.resetBasePositionAndOrientation.assert_called_once_with(
        7,
        (1.0, 2.0, 0.05),
        "quaternion",
    )
    pybullet.resetBaseVelocity.assert_called_once_with(
        7,
        linearVelocity=(0.0, 0.0, 0.0),
        angularVelocity=(0.0, 0.0, 0.0),
    )
    assert pybullet.resetJointState.call_args_list == [
        call(7, 0, targetValue=0.0, targetVelocity=0.0),
        call(7, 1, targetValue=0.0, targetVelocity=0.0),
    ]
    assert pybullet.setJointMotorControl2.call_args_list == [
        call(7, 0, pybullet.VELOCITY_CONTROL, targetVelocity=0.0, force=0.0),
        call(7, 1, pybullet.VELOCITY_CONTROL, targetVelocity=0.0, force=0.0),
    ]
