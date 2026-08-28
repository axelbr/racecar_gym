from unittest.mock import MagicMock

import gymnasium

import numpy as np

from racecar_gym.bullet.sensors import AccelerationSensor, FixedTimestepSensor


def test_fixed_timestep_sensor_discards_cached_observation_on_reset():
    sensor = MagicMock()
    sensor.name = "sensor"
    sensor.type = "test"
    sensor.space.return_value = gymnasium.spaces.Box(0.0, 1.0, shape=(1,))
    sensor.observe.side_effect = ["first episode", "second episode"]
    fixed_timestep_sensor = FixedTimestepSensor(
        sensor=sensor,
        frequency=25,
        time_step=0.01,
    )

    assert fixed_timestep_sensor.observe() == "first episode"
    assert fixed_timestep_sensor.observe() == "first episode"

    fixed_timestep_sensor.reset(body_id=3, joint_index=4)

    assert fixed_timestep_sensor.observe() == "second episode"
    sensor.reset.assert_called_once_with(body_id=3, joint_index=4)


def test_acceleration_sensor_discards_previous_velocity_on_reset():
    sensor = AccelerationSensor(
        name="acceleration",
        type="acceleration",
        config=AccelerationSensor.Config(time_delta=0.01, gaussian_noise=0.0),
    )
    sensor._last_velocity = np.ones(6)

    sensor.reset(body_id=3)

    np.testing.assert_array_equal(sensor._last_velocity, np.zeros(6))
