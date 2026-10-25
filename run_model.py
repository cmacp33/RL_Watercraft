#!/usr/bin/env python3
"""Run a trained pool policy in Gazebo without exploration noise."""
import argparse
import math
import os
import sys
import time

import gym
import gym_gazebo
import numpy as np
import torch

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_DIR, 'training_scripts'))
from training_scripts.gazebo_pool_v0_xentropy import Net, HIDDEN_SIZE, OBS_SIZE, ACT_SIZE

GOAL_RADIUS_M = 0.5 # matches done_check in pool_gym_v0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', default=os.path.join(REPO_DIR, 'saved_weights', 'pool_policy.pth'))
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--goal', type=float, nargs=2, metavar=('X', 'Y'),
                        help='override env goal in tank coordinates (m)')
    parser.add_argument('--noise', type=float, default=0.0,
                        help='optional action noise std, 0 = deterministic policy')
    return parser.parse_args()


def choose_action(net, obs, noise_std):
    with torch.no_grad():
        action = torch.tanh(net(torch.FloatTensor([obs]))).numpy()[0]
    if noise_std > 0:
        action = action + np.random.normal(0, noise_std, size=ACT_SIZE)
    return np.clip(action, -1.0, 1.0)


if __name__ == '__main__':
    args = parse_args()

    # load policy
    net = Net(OBS_SIZE, HIDDEN_SIZE, ACT_SIZE)
    net.load_state_dict(torch.load(args.weights, map_location='cpu'))
    net.eval()
    print("Loaded weights from", args.weights)

    # setup env
    env = gym.make('GazeboPool-v0')
    time.sleep(2)
    pool_env = env.unwrapped

    # goal is read in get_state, so set it before reset
    if args.goal is not None:
        pool_env.goal = np.array(args.goal, dtype=np.float32)
    print("Goal:", pool_env.goal)

    results = []
    for episode in range(args.episodes):
        obs = env.reset()
        start_dist = pool_env.position_scale_m * math.hypot(obs[0], obs[1])
        episode_reward = 0.0
        step_count = 0
        done = False

        while not done:
            action = choose_action(net, obs, args.noise)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            step_count += 1

            dist = pool_env.position_scale_m * math.hypot(obs[0], obs[1])
            heading_err_deg = math.degrees(math.atan2(obs[2], obs[3]))
            print("Ep %d step %3d | action [%+.2f %+.2f] | dist %.2f m | heading err %+6.1f deg | reward %+.3f" % (
                episode, step_count, action[0], action[1], dist, heading_err_deg, reward))

        final_dist = pool_env.position_scale_m * math.hypot(obs[0], obs[1])
        reached = final_dist <= GOAL_RADIUS_M
        results.append((episode_reward, final_dist, reached, step_count))
        print("Episode %d: total reward=%.1f, start dist=%.2f m, final dist=%.2f m, steps=%d, %s" % (
            episode, episode_reward, start_dist, final_dist, step_count,
            "REACHED GOAL" if reached else "timed out"))

    # summary
    rewards = [r[0] for r in results]
    final_dists = [r[1] for r in results]
    n_reached = sum(r[2] for r in results)
    print("\n=== %d episodes ===" % len(results))
    print("reached goal: %d/%d" % (n_reached, len(results)))
    print("reward mean=%.1f min=%.1f max=%.1f" % (np.mean(rewards), np.min(rewards), np.max(rewards)))
    print("final dist mean=%.2f m" % np.mean(final_dists))
