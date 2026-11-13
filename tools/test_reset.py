#!/usr/bin/env python3
"""Run random actions for 25 wall-clock seconds, then verify world reset.

Run on a machine with the catkin workspace sourced:
    python3 training_scripts/gazebo_pool_v0_reset_test.py
This checks the environment lifecycle; it does not train a policy.
"""
from watercraft.config import load
import math
import time

import gym
import gym_gazebo  # Registers GazeboPool-v0.
import rospy
from gazebo_msgs.srv import GetModelState


def boat_state(get_state):
    result = get_state('boat', 'world')
    if not result.success:
        raise RuntimeError(result.status_message)
    return result


def main():
    # Exercise one timed rollout even if the temporary task signals done early.
    env = gym.make(load('sim')['environment_id']).unwrapped
    rospy.wait_for_service('/gazebo/get_model_state', timeout=15)
    get_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
    env.reset()
    original = boat_state(get_state)
    print('Initial boat pose:', original.pose, flush=True)
    start = time.monotonic()
    steps = 0
    try:
        while time.monotonic() - start < load("tools")["sim_test_seconds"] and not rospy.is_shutdown():
            observation, reward, done, info = env.step(env.action_space.sample())
            steps += 1
            print('Step {} | elapsed {:.2f}s | reward {}'.format(
                steps, time.monotonic() - start, reward), flush=True)
        print('Before reset:', boat_state(get_state).pose, flush=True)
    finally:
        # Preserve the existing diagnostic; explicitly neutralize the command first.
        env.left_pub.publish(0.0)
        env.right_pub.publish(0.0)
        env.reset()

    restored = boat_state(get_state)
    p0, p1 = original.pose.position, restored.pose.position
    distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(
        (p0.x, p0.y, p0.z), (p1.x, p1.y, p1.z))))
    q0, q1 = original.pose.orientation, restored.pose.orientation
    dot = abs(q0.x*q1.x + q0.y*q1.y + q0.z*q1.z + q0.w*q1.w)
    angle = 2 * math.acos(min(1.0, dot))
    v, w = restored.twist.linear, restored.twist.angular
    speed = math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z)
    angular_speed = math.sqrt(w.x*w.x + w.y*w.y + w.z*w.z)
    print('After reset:', restored.pose, flush=True)
    print('Reset errors: position {:.4f}m, orientation {:.4f}rad; '
          'speed {:.4f}m/s, angular speed {:.4f}rad/s'.format(
              distance, angle, speed, angular_speed), flush=True)
    if max(distance, angle, speed, angular_speed) > load("tools")["reset_tolerance"]:
        raise RuntimeError('Reset verification failed (tolerance 0.1).')
    print('PASS: boat restored; simulation remains paused.', flush=True)


if __name__ == '__main__':
    main()
