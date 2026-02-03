#!/usr/bin/env python3
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

# params
HIDDEN_SIZE = 128
BATCH_SIZE = 8
PERCENTILE = 70
REWARD_THRESHOLD = 150
OBS_SIZE = 6
ACT_SIZE = 2


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
Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action']) 


# manage env to get batch data
def iterate_batches(env, net, batch_size):

    # init parameters
    batch = []
    episode_steps = []
    episode_reward = 0.0
    obs = env.reset()

    while True:

        # breakout actions and obsevations
        scale = 0.01
        obs_v = torch.FloatTensor([obs])
        action_v = net(obs_v)
        action = scale * action_v.data.numpy()[0]
        action = np.tanh(action)

        # step
        next_obs, reward, done, _ = env.step(action)
        episode_reward += reward

        # add to episode step history
        episode_steps.append(EpisodeStep(observation=obs, action=action))

        print("Reward:", episode_reward, " action:", action, "observation:", obs_v)

        # check if done, if done reset the env
        if done:
            # add episode to history and reset everything
            batch.append(Episode(reward=episode_reward, steps=episode_steps))
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()

            if len(batch) == batch_size:
                yield batch
                batch = []
        
        obs = next_obs
        time.sleep(0.1)

# filter batches to gind the best ones
def filter_batch(batch, percentile):

    rewards = list(map(lambda s: s.reward, batch))
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


if __name__ == '__main__':

    # setup env
    env = gym.make('GazeboPool-v0')
    time.sleep(2)
    obs_size = OBS_SIZE
    n_actions = ACT_SIZE

    # set nn
    net = Net(obs_size, HIDDEN_SIZE, n_actions)

    optimizer = optim.Adam(params=net.parameters(), lr=0.01)

    # handle batches
    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):

        #filter batches
        obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)
        optimizer.zero_grad()
        action_scores_v = net(obs_v)

        # compute losses and back prop
        loss_v = nn.MSELoss()(action_scores_v, acts_v)
        loss_v.backward()
        optimizer.step()

        print("%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f" % (
            iter_no, loss_v.item(), reward_m, reward_b))

        # stop if solved
        if reward_m > REWARD_THRESHOLD:
            print("Solved!")
            break
