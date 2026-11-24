"""Goal hold: the goal counts as reached only after goal_hold_steps steps in a row inside the radius.

Exercises the environment's counting and reward logic directly, without starting Gazebo.
"""
import unittest

import numpy as np

try:
    import gym_gazebo.envs.pool_gym_env.pool_gym_v0 as implementation
except ImportError as error:  # ROS (rospy, gazebo_msgs) is not installed here
    implementation = None
    IMPORT_ERROR = error

RADIUS_M = 0.08


def make_env(hold_steps, max_steps=300):
    env = implementation.GazeboPoolv0Env.__new__(implementation.GazeboPoolv0Env)
    implementation.GOAL_RADIUS_M = RADIUS_M
    env.position_scale_m = 1.0
    env.max_episode_steps = max_steps
    env.distance_weight = env.heading_weight = env.spin_weight = env.step_penalty = 0.0
    env.goal_bonus = 30.0
    env.goal_hold_steps = hold_steps
    env.hold_count = env.best_hold = env.count = 0
    return env


def run(env, distances):
    """Feed a sequence of goal distances; return (rewards, step at which the episode ended or None)."""
    rewards, previous = [], np.array([distances[0], 0, 0, 1, 0, 0], dtype=np.float32)
    for step, distance in enumerate(distances, start=1):
        state = np.array([distance, 0, 0, 1, 0, 0], dtype=np.float32)
        env.update_hold(state)
        rewards.append(env.compute_reward(state, previous))
        env.count += 1
        if env.done_check(state, previous):
            return rewards, step
        previous = state
    return rewards, None


@unittest.skipIf(implementation is None, 'ROS environment not importable')
class GoalHoldTests(unittest.TestCase):
    def test_hold_of_one_is_the_old_first_touch_behaviour(self):
        env = make_env(1)
        rewards, ended = run(env, [0.5, 0.3, 0.05, 0.05])
        self.assertEqual(ended, 3)
        self.assertEqual(rewards, [0.0, 0.0, 30.0])
        self.assertTrue(env.goal_held())

    def test_touching_the_goal_does_not_end_the_episode(self):
        env = make_env(5)
        rewards, ended = run(env, [0.5, 0.05, 0.05, 0.5, 0.5])
        self.assertIsNone(ended)
        self.assertFalse(env.goal_held())
        self.assertEqual(env.hold_count, 0)  # leaving the radius restarts the count

    def test_episode_ends_after_holding_and_pays_the_whole_bonus(self):
        env = make_env(5)
        rewards, ended = run(env, [0.5] + [0.05] * 10)
        self.assertEqual(ended, 6)
        self.assertAlmostEqual(sum(rewards), 30.0)
        self.assertTrue(env.goal_held())

    def test_leaving_and_returning_never_pays_more_than_the_bonus(self):
        env = make_env(5)
        dip = [0.05] * 4 + [0.5]  # four steps inside, then out: never completes the hold
        rewards, ended = run(env, dip * 20)
        self.assertIsNone(ended)
        self.assertAlmostEqual(sum(rewards), 30.0 * 4 / 5)
        rewards, ended = run(env, [0.05] * 5)  # now hold properly
        self.assertEqual(ended, 5)
        self.assertAlmostEqual(sum(rewards), 30.0 / 5)  # only the one missing part

    def test_timeout_inside_the_radius_is_not_reached(self):
        env = make_env(5, max_steps=3)
        rewards, ended = run(env, [0.05] * 3)
        self.assertEqual(ended, 3)
        self.assertFalse(env.goal_held())


if __name__ == '__main__':
    unittest.main()
