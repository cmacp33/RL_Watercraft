import math
import unittest

import numpy as np

from policy_contract import (
    NO_CALIBRATION,
    MotorCalibration,
    PoseObservationAdapter,
    esp32_motor_action,
    gazebo_thrust_action,
    policy_observation,
)


class PolicyContractTests(unittest.TestCase):
    def test_goal_ahead_has_zero_lateral_and_heading_error(self):
        observation = policy_observation(0, 0, 0, 0, 0, 0, 3, 0)

        np.testing.assert_allclose(observation, [3 / 20, 0, 0, 1, 0, 0])

    def test_goal_to_left_is_positive_left_and_heading_error(self):
        observation = policy_observation(0, 0, 0, 0, 0, 0, 0, 3)

        self.assertGreater(observation[1], 0)
        self.assertGreater(observation[2], 0)

    def test_forward_velocity_is_body_relative(self):
        observation = policy_observation(
            0, 0, math.pi / 2, 0, 1, 0, 0, 3
        )

        self.assertAlmostEqual(float(observation[4]), 1 / 3, places=6)

    def test_actions_map_to_platform_ranges(self):
        action = [-0.5, 1.2]

        np.testing.assert_allclose(gazebo_thrust_action(action), [-0.5, 1.0])
        np.testing.assert_array_equal(
            esp32_motor_action(action, calibration=NO_CALIBRATION), [-128, 255])

    def test_motor_calibration_gain_clamp_and_swap(self):
        cal = MotorCalibration(swap_channels=True, left_gain=1.8, right_gain=1.0)
        # physical left 0.4 * 1.8 -> 184, physical right 0.4 -> 102; swapped onto ESP channels
        np.testing.assert_array_equal(esp32_motor_action([0.4, 0.4], calibration=cal), [102, 184])
        # gain can't push past full scale
        np.testing.assert_array_equal(esp32_motor_action([1.0, -1.0], calibration=cal), [-255, 255])

    def test_motor_calibration_usable_range(self):
        cal = MotorCalibration(deadband=40, max_command=90, min_action=0.05)
        # tiny outputs stay off instead of jumping to the deadband
        np.testing.assert_array_equal(esp32_motor_action([0.0, 0.04], calibration=cal), [0, 0])
        # just above min_action starts at the deadband; full scale stops at max_command
        np.testing.assert_array_equal(esp32_motor_action([0.06, -0.5], calibration=cal), [43, -65])
        np.testing.assert_array_equal(esp32_motor_action([-1.0, 1.0], calibration=cal), [-90, 90])

    def test_real_pose_adapter_estimates_velocity_and_wraps_yaw(self):
        adapter = PoseObservationAdapter(position_scale_m=0.5)
        adapter.update(0, 0, 179, 1.0, 25, 0)
        observation = adapter.update(10, 0, -179, 2.0, 25, 0)

        self.assertAlmostEqual(float(observation[4]), -1 / 30, places=4)
        self.assertAlmostEqual(float(observation[5]), math.radians(2) / 2.0, places=6)


if __name__ == "__main__":
    unittest.main()