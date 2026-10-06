import math
import unittest

import numpy as np

from policy_contract import (
    PoseObservationAdapter,
    esp32_motor_action,
    gazebo_thrust_action,
    policy_observation,
)


class PolicyContractTests(unittest.TestCase):
    def test_goal_ahead_has_zero_lateral_and_heading_error(self):
        observation = policy_observation(0, 0, 0, 0, 0, 0, 3, 0)

        np.testing.assert_allclose(observation, [0.2, 0, 0, 1, 0, 0])

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
        np.testing.assert_array_equal(esp32_motor_action(action), [-128, 255])

    def test_real_pose_adapter_estimates_velocity_and_wraps_yaw(self):
        adapter = PoseObservationAdapter(position_scale_m=0.5)
        adapter.update(0, 0, 179, 1.0, 25, 0)
        observation = adapter.update(10, 0, -179, 2.0, 25, 0)

        self.assertAlmostEqual(float(observation[4]), -1 / 30, places=4)
        self.assertAlmostEqual(float(observation[5]), math.radians(2) / 2.0, places=6)


if __name__ == "__main__":
    unittest.main()