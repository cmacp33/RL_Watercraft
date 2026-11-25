"""policy.heading_mode 'target': observations 2-3 are the error to the goal heading, not the bearing to the goal."""
import math
import unittest

import numpy as np

from watercraft.contract import policy_observation


def observe(x, y, yaw_deg, goal, goal_heading_deg=None):
    heading = None if goal_heading_deg is None else math.radians(goal_heading_deg)
    return policy_observation(x, y, math.radians(yaw_deg), 0.0, 0.0, 0.0, *goal,
                              position_scale_m=1.0, goal_heading_rad=heading)


class GoalHeadingObservationTests(unittest.TestCase):
    def test_without_a_goal_heading_the_observation_is_unchanged(self):
        # goal straight to the left of a boat facing +x: bearing error +90 degrees
        np.testing.assert_allclose(observe(0, 0, 0, (0, 0.5)), [0, 0.5, 1, 0, 0, 0], atol=1e-6)

    def test_heading_error_is_to_the_goal_heading_not_the_goal(self):
        # same goal to the left, but the goal heading is +x, which the boat already faces
        np.testing.assert_allclose(observe(0, 0, 0, (0, 0.5), goal_heading_deg=0), [0, 0.5, 0, 1, 0, 0], atol=1e-6)

    def test_heading_error_sign_is_counterclockwise_positive(self):
        obs = observe(0, 0, 0, (0.3, 0), goal_heading_deg=90)  # must turn left (counterclockwise)
        self.assertAlmostEqual(math.degrees(math.atan2(obs[2], obs[3])), 90.0, places=4)
        obs = observe(0, 0, 0, (0.3, 0), goal_heading_deg=-45)
        self.assertAlmostEqual(math.degrees(math.atan2(obs[2], obs[3])), -45.0, places=4)

    def test_heading_error_wraps_across_180_degrees(self):
        obs = observe(0, 0, 170, (0.3, 0), goal_heading_deg=-170)  # 20 degrees counterclockwise, not 340 clockwise
        self.assertAlmostEqual(math.degrees(math.atan2(obs[2], obs[3])), 20.0, places=4)

    def test_heading_error_is_steady_at_the_goal(self):
        # 2 mm of position noise at the goal: the bearing swings, the goal-heading error does not
        bearing = [math.degrees(math.atan2(*observe(dx, dy, 0, (0, 0))[2:4])) for dx, dy in ((0.002, 0), (-0.002, 0))]
        target = [math.degrees(math.atan2(*observe(dx, dy, 0, (0, 0), 30)[2:4])) for dx, dy in ((0.002, 0), (-0.002, 0))]
        self.assertGreater(abs(bearing[0] - bearing[1]), 170)
        self.assertAlmostEqual(target[0], target[1], places=6)

    def test_non_finite_goal_heading_is_rejected(self):
        with self.assertRaises(ValueError):
            observe(0, 0, 0, (0.3, 0), goal_heading_deg=float('nan'))


if __name__ == '__main__':
    unittest.main()
