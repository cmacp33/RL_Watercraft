#!/usr/bin/env python3
import time
from boat_env import BoatEnv

# init environment
env = BoatEnv()

# reset environment
state = env.reset()

done = False
step_count = 0

while not done:

    action = env.action_space.sample()
    state, reward, done = env.step(action[0], action[1])
    print(
        f"Step {step_count} | Action: {action} | "
        f"State: {state} | Reward: {reward} | Done: {done}"
    )

    step_count += 1
    time.sleep(0.5)


