"""Compatibility entry point. Settings: configs/train.json and configs/algorithms/cem.json."""
from types import SimpleNamespace
from watercraft.algorithms.cem import *
from watercraft.config import load
from train import main

_ALGORITHM = load('algorithms/cem')
_TRAINING = load('train')
BATCH_SIZE = _ALGORITHM['batch_size']
PERCENTILE = _ALGORITHM['percentile']
STOP_SUCCESS_RATE = _ALGORITHM['stop_success']
STOP_SUCCESS_ITERS = _ALGORITHM['stop_iterations']
MAX_TRAIN_SEC = _TRAINING['minutes'] * 60
NOISE_START = _ALGORITHM['noise_start']
NOISE_END = _ALGORITHM['noise_end']
NOISE_DECAY_ITERS = _ALGORITHM['noise_decay_iterations']
ELITE_MEMORY = _ALGORITHM['elite_memory']
TRAIN_STEPS_PER_ITER = _ALGORITHM['train_steps_per_iteration']
LR = _ALGORITHM['learning_rate']
from watercraft.config import ROOT
LATEST_WEIGHTS = str(ROOT / 'saved_weights/pool_policy.pth')

def parse_args():
    training, algorithm, sim = load('train'), load('algorithms/cem'), load('sim')
    return SimpleNamespace(tag=training['run'], minutes=training['minutes'],
        stop_success=algorithm['stop_success'], stop_iters=algorithm['stop_iterations'],
        random_goals=sim['random_goals'], batch_size=algorithm['batch_size'], randomize=sim['domain_randomization'])

if __name__ == '__main__':
    main()
