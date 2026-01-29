#!/usr/bin/env python3
import time
import gym
from gym import wrappers
import gym_gazebo
import time
import numpy
import random

if __name__ == '__main__':
    # init environment
    env = gym.make('GazeboPool-v0')
    time.sleep(2)

    total_episodes = 10000

    print("*** Done setting environment.")

    for x in range(total_episodes):

        done = False

        # reset environment
        state = env.reset()
        step_count = 0

        while True:
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)

            print(
                f"Step {step_count} | Action: {action} | "
                f"State: {state} | Reward: {reward} | Done: {done}"
            )

            step_count += 1
            time.sleep(0.5)

