"""Per-run result logging so different reward functions can be compared.

Each training run gets its own folder under <repo>/runs/:
    config.json        hyperparameters, reward weights, goal, git commit
    reward_function.py exact source of compute_reward/done_check used
    episodes.csv       one row per training episode
    iterations.csv     one row per training iteration (batch)
    best.pth/final.pth policy weights
    eval.csv           written by run_model.py --run <dir>
    events.out.tfevents.*  TensorBoard curves: tensorboard --logdir runs

Total reward is NOT comparable between reward functions (different scales),
so every row also logs reward-independent task metrics: whether the goal was
reached, final/closest distance, and steps taken.
"""
import csv
import inspect
import json
import math
import os
import subprocess
import time

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(REPO_DIR, 'runs')

EPISODE_FIELDS = ['iter', 'episode', 'total_reward', 'steps', 'start_dist_m',
                  'final_dist_m', 'min_dist_m', 'reached', 'goal_x_m', 'goal_y_m', 'score', 'noise_std',
                  'elapsed_s']
ITERATION_FIELDS = ['iter', 'elapsed_s', 'loss', 'reward_mean', 'reward_bound',
                    'success_rate', 'mean_final_dist_m', 'mean_min_dist_m',
                    'mean_steps', 'train_steps_kept', 'noise_std', 'score_mean', 'score_bound',
                    'new_elites', 'memory_size', 'memory_success']


def goal_distance_m(env, obs):
    """Distance to goal in metres, recovered from the normalized observation."""
    return env.position_scale_m * math.hypot(obs[0], obs[1])


def _git(*args):
    try:
        return subprocess.check_output(['git', '-C', REPO_DIR] + list(args),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def make_run_dir(tag):
    safe_tag = ''.join(c if c.isalnum() or c in '-_' else '-' for c in tag)
    run_dir = os.path.join(RUNS_DIR, time.strftime('%Y%m%d-%H%M%S') + '_' + safe_tag)
    os.makedirs(run_dir)
    return run_dir


def write_config(run_dir, tag, env, hyperparams):
    reward_params = {name: getattr(env, name) for name in
                     ('distance_weight', 'heading_weight', 'step_penalty', 'goal_bonus')
                     if hasattr(env, name)}
    config = {
        'tag': tag,
        'started': time.strftime('%Y-%m-%d %H:%M:%S'),
        'git_commit': _git('rev-parse', '--short', 'HEAD'),
        'git_dirty': bool(_git('status', '--porcelain', '--', '*.py')),
        'goal': 'random' if getattr(env, 'random_goals', False) else [float(v) for v in env.goal],
        'goal_distance_range_m': (list(env.goal_distance_range_m)
                                  if getattr(env, 'random_goals', False) else None),
        'position_scale_m': env.position_scale_m,
        'max_episode_steps': getattr(env, 'max_episode_steps', None),
        'domain_randomization': (getattr(env, 'dr_ranges', None)
                                 if getattr(env, 'domain_randomization', False) else False),
        'control_period_s': getattr(env, 'control_period_s', None),
        'reward_params': reward_params,
        'hyperparams': hyperparams,
    }
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # the weights alone don't capture the reward's shape, so keep its exact code
    env_cls = type(env)
    with open(os.path.join(run_dir, 'reward_function.py'), 'w') as f:
        f.write('# Source of %s at run start\n\n' % env_cls.__name__)
        for method in ('compute_reward', 'done_check'):
            f.write(inspect.getsource(getattr(env_cls, method)) + '\n')


def finish_run(run_dir, stop_reason, total_goals, total_episodes):
    """Record how the run ended in config.json."""
    path = os.path.join(run_dir, 'config.json')
    with open(path) as f:
        config = json.load(f)
    config.update(finished=time.strftime('%Y-%m-%d %H:%M:%S'), stop_reason=stop_reason,
                  total_goals=total_goals, total_episodes=total_episodes)
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)


class TensorBoardLog:
    """Mirror per-iteration metrics to TensorBoard; a no-op if it isn't installed."""

    # metric -> chart name; grouped so related curves sit together in TensorBoard
    CHARTS = {
        'success_rate': 'task/success_rate',
        'mean_final_dist_m': 'task/final_dist_m',
        'mean_min_dist_m': 'task/closest_dist_m',
        'mean_steps': 'task/episode_steps',
        'reward_mean': 'reward/mean',
        'reward_bound': 'reward/bound',
        'loss': 'train/loss',
        'train_steps_kept': 'train/steps_kept',
        'score_mean': 'score/mean',
        'score_bound': 'score/elite_bound',
        'memory_success': 'score/memory_success_rate',
        'noise_std': 'train/noise_std',
    }

    def __init__(self, run_dir):
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(log_dir=run_dir)
        except Exception as error:  # missing tensorboard package, etc.
            print("TensorBoard logging disabled:", error)
            self._writer = None

    def write(self, step, row):
        if self._writer is None:
            return
        for key, chart in self.CHARTS.items():
            if key in row:
                self._writer.add_scalar(chart, row[key], step)
        self._writer.flush()

    def close(self):
        if self._writer is not None:
            self._writer.close()


class CsvLog:
    """Append rows to a CSV, flushing each one so Ctrl-C loses nothing."""

    def __init__(self, path, fields):
        self.fields = fields
        self._file = open(path, 'w', newline='')
        self._writer = csv.DictWriter(self._file, fieldnames=fields)
        self._writer.writeheader()
        self._file.flush()

    def write(self, **row):
        self._writer.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                               for k, v in row.items()})
        self._file.flush()

    def close(self):
        self._file.close()
