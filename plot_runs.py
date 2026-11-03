#!/usr/bin/env python3
"""Plot training curves for one or more runs in runs/, one line per run.

Usage:
    python3 plot_runs.py                          # all runs -> runs/compare.png
    python3 plot_runs.py runs/*300steps* --x minutes
    python3 plot_runs.py --smooth 5 --out figs/reward_comparison.png

Each batch is only 8 episodes, so per-batch success is coarse (steps of 12.5%).
The faint line is the raw value; the bold line is a rolling mean over --smooth batches.
Reward is plotted for reference only: it is on a different scale per reward function.
"""
import argparse
import csv
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')  # write files; no display needed
import matplotlib.pyplot as plt
import numpy as np

REPO_DIR = os.path.dirname(os.path.abspath(__file__))

PANELS = [
    ('success_rate', 'Success rate (reached goal)', 100.0, '%'),
    ('mean_final_dist_m', 'Final distance to goal', 1.0, 'm'),
    ('mean_min_dist_m', 'Closest approach to goal', 1.0, 'm'),
    ('mean_steps', 'Steps per episode', 1.0, 'steps'),
    ('reward_mean', 'Mean episode reward (not comparable across rewards)', 1.0, ''),
    ('loss', 'Training loss', 1.0, ''),
]


def rolling_mean(values, window):
    if window <= 1 or len(values) < 2:
        return values
    out = np.empty_like(values)
    for i in range(len(values)):
        out[i] = values[max(0, i - window + 1):i + 1].mean()
    return out


def load_run(run_dir):
    path = os.path.join(run_dir, 'iterations.csv')
    if not os.path.exists(path):
        return None
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    data = {key: np.array([float(r[key]) for r in rows]) for key in rows[0]}
    config_path = os.path.join(run_dir, 'config.json')
    config = json.load(open(config_path)) if os.path.exists(config_path) else {}
    data['label'] = '%s (%s)' % (os.path.basename(run_dir.rstrip('/')), config.get('stop_reason', 'running'))
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('runs', nargs='*', help='run folders (default: all of runs/)')
    parser.add_argument('--x', choices=['iter', 'minutes'], default='iter',
                        help='x axis: training iteration (fair across machines) or wall-clock minutes')
    parser.add_argument('--smooth', type=int, default=3, help='rolling-mean window in batches (1 = off)')
    parser.add_argument('--out', default=os.path.join(REPO_DIR, 'runs', 'compare.png'))
    args = parser.parse_args()

    run_dirs = args.runs or sorted(glob.glob(os.path.join(REPO_DIR, 'runs', '*')))
    runs = [r for r in (load_run(d) for d in run_dirs if os.path.isdir(d)) if r is not None]
    if not runs:
        print("No runs with iterations.csv found.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, run in enumerate(runs):
        color = colors[i % len(colors)]
        x = run['elapsed_s'] / 60 if args.x == 'minutes' else run['iter']
        for ax, (key, title, scale, unit) in zip(axes.flat, PANELS):
            y = run[key] * scale
            if args.smooth > 1:
                ax.plot(x, y, color=color, alpha=0.25, linewidth=1)
            ax.plot(x, rolling_mean(y, args.smooth), color=color, linewidth=2,
                    label=run['label'] if key == PANELS[0][0] else None)

    for ax, (key, title, scale, unit) in zip(axes.flat, PANELS):
        ax.set_title(title)
        ax.set_xlabel('training iteration' if args.x == 'iter' else 'minutes')
        if unit:
            ax.set_ylabel(unit)
        ax.grid(alpha=0.3)
    axes.flat[0].set_ylim(-5, 105)
    fig.legend(loc='lower center', ncol=min(3, len(runs)), fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.05 + 0.03 * ((len(runs) - 1) // 3), 1, 1))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print("Saved", args.out)


if __name__ == '__main__':
    main()
