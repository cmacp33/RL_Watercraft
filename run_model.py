"""Compatibility evaluation entry point; settings are in configs/evaluate.json."""
import numpy as np
import torch
from types import SimpleNamespace
from watercraft.algorithms.cem import Net, HIDDEN_SIZE, OBS_SIZE, ACT_SIZE
from watercraft.config import load
from evaluate import main, EVAL_FIELDS

def parse_args():
    return SimpleNamespace(**load('evaluate'))

def choose_action(net, obs, noise_std):
    with torch.no_grad():
        action = torch.tanh(net(torch.FloatTensor([obs]))).numpy()[0]
    if noise_std > 0:
        action = action + np.random.normal(0, noise_std, size=ACT_SIZE)
    return np.clip(action, -1.0, 1.0)


if __name__ == '__main__':
    main()
