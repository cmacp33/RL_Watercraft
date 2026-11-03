#!/usr/bin/env python3
import argparse
import os
import shutil
import time
import gym
from gym import wrappers
import gym_gazebo
import numpy as np
import random
from collections import namedtuple

# torch libs
import torch
import torch.nn as nn
import torch.optim as optim

from policy_contract import GOAL_RADIUS_M
from export_policy import export as export_policy
from run_logging import (EPISODE_FIELDS, ITERATION_FIELDS, REPO_DIR, CsvLog, TensorBoardLog, finish_run,
                         goal_distance_m, make_run_dir, write_config)

# params
HIDDEN_SIZE = 128
BATCH_SIZE = 8 # number of episodes
PERCENTILE = 70
# stop early once the policy reliably reaches the goal: success rate >= this
# for this many consecutive batches (task-based, so it works for any reward)
STOP_SUCCESS_RATE = 0.9
STOP_SUCCESS_ITERS = 3
OBS_SIZE = 6
ACT_SIZE = 2

MAX_TRAIN_SEC = 60 * 60 # 1 hour of training

# noise standard deviation
NOISE_STD = 0.5

# absolute so the result doesn't depend on which folder the script is run from
LATEST_WEIGHTS = os.path.join(REPO_DIR, 'saved_weights', 'pool_policy.pth')

# define the nn class
class Net(nn.Module):

    # init
    def __init__(self, obs_size, hidden_size, n_actions):

        super(Net, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)


# save episode and step history
Episode = namedtuple('Episode', field_names=['reward', 'steps', 'start_dist', 'final_dist',
                                             'min_dist', 'reached'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])


# manage env to get batch data
def iterate_batches(env, net, batch_size):

    pool_env = env.unwrapped

    # init parameters
    batch = []
    episode_steps = []
    episode_reward = 0.0
    sim_dts = []
    obs = env.reset()
    start_dist = min_dist = pool_env.goal_distance_m()  # true distance, not the noisy obs

    while True:

        # choose an action
        obs_v = torch.FloatTensor([obs])
        action_v = net(obs_v)
        action = action_v.data.numpy()[0]
        action = np.clip(np.tanh(action) + np.random.normal(0, NOISE_STD, size=ACT_SIZE), -1.0, 1.0)

        # step
        next_obs, reward, done, info = env.step(action)
        sim_dts.append(info.get('sim_dt', float('nan')))
        episode_reward += reward
        dist = info['goal_dist_m']
        min_dist = min(min_dist, dist)

        # add to episode step history
        episode_steps.append(EpisodeStep(observation=obs, action=action))

        # check if done, if done reset the env
        if done:
            # add episode to history and reset everything
            batch.append(Episode(reward=episode_reward, steps=episode_steps,
                                 start_dist=start_dist, final_dist=dist, min_dist=min_dist,
                                 reached=dist <= GOAL_RADIUS_M))

            # reset reward, steps, and env
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()
            start_dist = min_dist = pool_env.goal_distance_m()

            if len(batch) == batch_size:
                yield batch, float(np.mean(sim_dts))
                batch = []
                sim_dts = []

        obs = next_obs

# filter batches to gind the best ones
def filter_batch(batch, percentile):

    rewards = [episode.reward for episode in batch] # lists rewards of every batch
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))

    train_obs = []
    train_act = []

    for example in batch:
        if example.reward < reward_bound:
            continue

        train_obs.extend(map(lambda step: step.observation, example.steps))
        train_act.extend(map(lambda step: step.action, example.steps))

    train_obs_v = torch.FloatTensor(train_obs)
    train_act_v = torch.FloatTensor(np.array(train_act, dtype=np.float32))
    return train_obs_v, train_act_v, reward_bound, reward_mean


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-entropy training for GazeboPool-v0")
    parser.add_argument('--tag', default='xentropy',
                        help='short label for this run, e.g. the reward variant being tested')
    parser.add_argument('--minutes', type=float, default=MAX_TRAIN_SEC / 60,
                        help='wall-clock training limit')
    parser.add_argument('--stop-success', type=float, default=STOP_SUCCESS_RATE,
                        help='stop once batch success rate is >= this (set above 1 to disable)')
    parser.add_argument('--stop-iters', type=int, default=STOP_SUCCESS_ITERS,
                        help='... for this many consecutive batches')
    parser.add_argument('--randomize', action='store_true',
                        help='domain randomization: random motor gains/deadband/latency + sensor noise per episode')
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    # setup env
    env = gym.make('GazeboPool-v0') # init gym registered under gym_gazebo __init__
    time.sleep(2)
    pool_env = env.unwrapped
    pool_env.domain_randomization = args.randomize
    obs_size = OBS_SIZE
    n_actions = ACT_SIZE

    # set nn
    net = Net(obs_size, HIDDEN_SIZE, n_actions)

    optimizer = optim.Adam(params=net.parameters(), lr=0.01)

    # results folder for this run
    run_dir = make_run_dir(args.tag)
    write_config(run_dir, args.tag, pool_env, {
        'algorithm': 'cross-entropy', 'hidden_size': HIDDEN_SIZE, 'batch_size': BATCH_SIZE,
        'percentile': PERCENTILE, 'lr': 0.01, 'noise_std': NOISE_STD,
        'max_minutes': args.minutes, 'stop_success_rate': args.stop_success,
        'stop_success_iters': args.stop_iters,
    })
    episode_log = CsvLog(os.path.join(run_dir, 'episodes.csv'), EPISODE_FIELDS)
    iteration_log = CsvLog(os.path.join(run_dir, 'iterations.csv'), ITERATION_FIELDS)
    tb_log = TensorBoardLog(run_dir)
    print("Logging this run to", run_dir)
    print("Live charts: tensorboard --logdir %s" % os.path.join(REPO_DIR, "runs"))

    start_time = time.monotonic()
    best_score = None
    episode_no = 0
    total_goals = 0
    success_streak = 0
    stop_reason = 'unknown'

    try:
        # handle batches
        for iter_no, (batch, mean_sim_dt) in enumerate(iterate_batches(env, net, BATCH_SIZE)):

            elapsed = time.monotonic() - start_time
            for ep in batch:
                episode_log.write(iter=iter_no, episode=episode_no, total_reward=ep.reward,
                                  steps=len(ep.steps), start_dist_m=ep.start_dist,
                                  final_dist_m=ep.final_dist, min_dist_m=ep.min_dist,
                                  reached=int(ep.reached), elapsed_s=elapsed)
                episode_no += 1
                total_goals += int(ep.reached)

            #filter batches
            obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE) # obs_v shape is (N,6)
            optimizer.zero_grad()
            action_scores_v = torch.tanh(net(obs_v))

            # compute losses and back prop
            loss_v = nn.MSELoss()(action_scores_v, acts_v)
            loss_v.backward()
            optimizer.step()

            success_rate = float(np.mean([ep.reached for ep in batch]))
            mean_final_dist = float(np.mean([ep.final_dist for ep in batch]))
            row = dict(iter=iter_no, elapsed_s=elapsed, loss=loss_v.item(),
                       reward_mean=reward_m, reward_bound=float(reward_b),
                       success_rate=success_rate, mean_final_dist_m=mean_final_dist,
                       mean_min_dist_m=float(np.mean([ep.min_dist for ep in batch])),
                       mean_steps=float(np.mean([len(ep.steps) for ep in batch])),
                       train_steps_kept=len(obs_v))
            iteration_log.write(**row)
            tb_log.write(iter_no, row)

            print("%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f, success=%.0f%%, final_dist=%.2f m, goals so far=%d/%d" % (
                iter_no, loss_v.item(), reward_m, reward_b, 100 * success_rate, mean_final_dist,
                total_goals, episode_no))
            print("   time: %.1f / %.0f min (%.1f min left at most), %.1f min per batch, sim step %.3f s" % (
                elapsed / 60, args.minutes, max(0.0, args.minutes - elapsed / 60),
                elapsed / 60 / (iter_no + 1), mean_sim_dt))
            if mean_sim_dt > 1.2 * pool_env.control_period_s:
                print("   WARNING: sim steps average %.3f s vs %.2f s intended; lower real_time_update_rate in boat.world"
                      % (mean_sim_dt, pool_env.control_period_s))

            # "best" uses task metrics, not reward, so it means the same thing for every reward function
            score = (success_rate, -mean_final_dist)
            if best_score is None or score > best_score:
                best_score = score
                torch.save(net.state_dict(), os.path.join(run_dir, 'best.pth'))

            # stop if solved
            success_streak = success_streak + 1 if success_rate >= args.stop_success else 0
            if success_streak >= args.stop_iters:
                stop_reason = 'solved'
                print("Solved: success >= %.0f%% for %d batches in a row" % (
                    100 * args.stop_success, args.stop_iters))
                break

            if time.monotonic() - start_time >= args.minutes * 60:
                stop_reason = 'time_limit'
                print("Training time limit reached")
                break
    except KeyboardInterrupt:
        stop_reason = 'interrupted'
        print("Interrupted, saving what we have")
    finally:
        finish_run(run_dir, stop_reason, total_goals, episode_no)
        episode_log.close()
        iteration_log.close()
        tb_log.close()
        torch.save(net.state_dict(), os.path.join(run_dir, 'final.pth'))
        os.makedirs(os.path.dirname(LATEST_WEIGHTS), exist_ok=True)
        shutil.copyfile(os.path.join(run_dir, 'final.pth'), LATEST_WEIGHTS)
        # numpy copies for the camera machine (no torch there): policy_contract.NumpyPolicy
        for name in ('best', 'final'):
            if os.path.exists(os.path.join(run_dir, name + '.pth')):
                export_policy(os.path.join(run_dir, name + '.pth'))
        print("Saved run to", run_dir, "and latest weights to", LATEST_WEIGHTS)
