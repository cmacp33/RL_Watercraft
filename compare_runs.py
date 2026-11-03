#!/usr/bin/env python3
"""Compare training runs in runs/ on reward-independent task metrics.

Usage:
    python3 compare_runs.py                 # all runs
    python3 compare_runs.py runs/2026*_v3*  # selected runs
    python3 compare_runs.py --last 10       # train metrics over the last 10 iterations

Columns:
    train_succ  goal-reached rate over the last N training iterations (noisy policy)
    train_dist  mean final distance (m) over those iterations
    goals       training episodes that reached the goal / total episodes
    first_goal  episode number of the first goal reached
    iters_50    first iteration whose batch reached the goal >= 50% of the time
    stop        why training ended: solved, time_limit or interrupted
    eval_*      from eval.csv (run_model.py --run <dir>): deterministic policy,
                the fairest comparison between reward functions
Total reward is shown only for reference: it is on a different scale per reward function.
"""
import argparse
import csv
import glob
import json
import os

REPO_DIR = os.path.dirname(os.path.abspath(__file__))


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def mean(rows, key):
    values = [float(r[key]) for r in rows]
    return sum(values) / len(values) if values else None


def fmt(value, spec):
    return '-' if value is None else format(value, spec)


def summarize(run_dir, last_n):
    config_path = os.path.join(run_dir, 'config.json')
    config = json.load(open(config_path)) if os.path.exists(config_path) else {}
    iters = read_csv(os.path.join(run_dir, 'iterations.csv'))
    evals = read_csv(os.path.join(run_dir, 'eval.csv'))
    episodes = read_csv(os.path.join(run_dir, 'episodes.csv'))
    goal_eps = [r for r in episodes if r['reached'] == '1']
    tail = iters[-last_n:]

    iters_50 = next((int(r['iter']) for r in iters if float(r['success_rate']) >= 0.5), None)
    params = config.get('reward_params', {})
    return {
        'run': os.path.basename(run_dir.rstrip('/')),
        'reward': ' '.join('%s=%g' % (k.split('_')[0][:4], v) for k, v in params.items()) or '-',
        'iters': str(len(iters)),
        'minutes': fmt(float(iters[-1]['elapsed_s']) / 60 if iters else None, '.0f'),
        'train_succ': fmt(mean(tail, 'success_rate'), '.0%'),
        'train_dist': fmt(mean(tail, 'mean_final_dist_m'), '.2f'),
        'goals': '%d/%d' % (len(goal_eps), len(episodes)),
        'first_goal': goal_eps[0]['episode'] if goal_eps else '-',
        'iters_50': '-' if iters_50 is None else str(iters_50),
        'stop': config.get('stop_reason', 'running?'),
        'eval_n': str(len(evals)) if evals else '-',
        'eval_succ': fmt(mean(evals, 'reached'), '.0%'),
        'eval_dist': fmt(mean(evals, 'final_dist_m'), '.2f'),
        'eval_steps': fmt(mean(evals, 'steps'), '.0f'),
        'reward_mean': fmt(mean(tail, 'reward_mean'), '.1f'),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('runs', nargs='*', help='run folders (default: all of runs/)')
    parser.add_argument('--last', type=int, default=5, help='iterations to average for train metrics')
    args = parser.parse_args()

    run_dirs = args.runs or sorted(glob.glob(os.path.join(REPO_DIR, 'runs', '*')))
    run_dirs = [d for d in run_dirs if os.path.isdir(d)]
    if not run_dirs:
        print("No runs found. Train with: python3 training_scripts/gazebo_pool_v0_xentropy.py --tag <name>")
        return

    rows = [summarize(d, args.last) for d in run_dirs]
    columns = list(rows[0])
    widths = {c: max(len(c), *(len(r[c]) for r in rows)) for c in columns}
    print('  '.join(c.ljust(widths[c]) for c in columns))
    for r in rows:
        print('  '.join(r[c].ljust(widths[c]) for c in columns))


if __name__ == '__main__':
    main()
