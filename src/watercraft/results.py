#!/usr/bin/env python3
"""Plot and compare configured runs, including historical CSV formats."""
import csv
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')  # write files; no display needed
import matplotlib.pyplot as plt
import numpy as np

REPO_DIR = str(__import__('watercraft.config', fromlist=['ROOT']).ROOT)

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
    config = json.loads(Path(config_path).read_text()) if os.path.exists(config_path) else {}
    summary = Path(run_dir) / 'summary.json'
    if summary.exists():
        config.update(json.loads(summary.read_text()))
    data['label'] = '%s (%s)' % (os.path.basename(run_dir.rstrip('/')), config.get('stop_reason', 'running'))
    return data


def render_training(run_dirs, output, x="minutes", smooth=3):
    from types import SimpleNamespace
    args = SimpleNamespace(runs=run_dirs, out=str(output), x=x, smooth=smooth)
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
    plt.close(fig)
    print("Saved", args.out)



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
    config = json.loads(Path(config_path).read_text()) if os.path.exists(config_path) else {}
    summary = Path(run_dir) / 'summary.json'
    if summary.exists():
        config.update(json.loads(summary.read_text()))
    iters = read_csv(os.path.join(run_dir, 'iterations.csv'))
    evals = read_csv(os.path.join(run_dir, 'eval.csv'))
    episodes = read_csv(os.path.join(run_dir, 'training.csv')) or read_csv(os.path.join(run_dir, 'episodes.csv'))
    goal_eps = [r for r in episodes if r['reached'] == '1']
    tail = iters[-last_n:]

    iters_50 = next((int(r['iter']) for r in iters if float(r['success_rate']) >= 0.5), None)
    params = config.get('reward_params', {})
    result = {
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
    # Report the latest simulation evaluation.
    for environment in ('sim',):
        files = sorted(Path(run_dir).glob(f'evaluations/{environment}/*.csv'))
        evaluated = read_csv(files[-1]) if files else []
        result[environment + '_eval_n'] = str(len(evaluated))
        result[environment + '_eval_success'] = fmt(mean(evaluated, 'reached'), '.0%')
        result[environment + '_eval_distance_m'] = fmt(mean(evaluated, 'final_dist_m'), '.2f')
        result[environment + '_eval_file'] = files[-1].name if files else '-'
    return result




from pathlib import Path
from watercraft.config import load, run_path, path


def selected_runs():
    names = load('results')['runs']
    return [run_path(name) for name in names] if names else sorted(p for p in path('runs').glob('*') if p.is_dir() and ((p / 'iterations.csv').exists() or (p / 'training.csv').exists()))


def plot_run(directory):
    settings = load('results')
    render_training([str(directory)], Path(directory) / 'graphs/training.png', settings['x'], settings['smooth'])


def plot_evaluation(filename):
    rows = read_csv(filename)
    if not rows:
        return
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    episodes = [int(r['episode']) for r in rows]
    axes[0].plot(episodes, [float(r['final_dist_m']) for r in rows], marker='o')
    axes[0].set(xlabel='Episode', ylabel='Final distance (m)')
    axes[1].bar(episodes, [int(r['reached']) for r in rows])
    axes[1].set(xlabel='Episode', ylabel='Reached goal', ylim=(0, 1.1))
    figure.tight_layout()
    output = Path(filename).with_suffix('.png')
    figure.savefig(output)
    plt.close(figure)


def plot_main():
    directories = selected_runs()
    for directory in directories:
        plot_run(directory)
        for filename in (directory / 'evaluations').glob('*/*.csv'):
            plot_evaluation(filename)
    if not directories:
        print('No runs with training metrics found')


def compare_main():
    settings = load('results')
    directories = selected_runs()
    rows = [summarize(str(d), settings['last_iterations']) for d in directories]
    if not rows:
        print('No runs found')
        return
    output = path(settings['comparison_output'])
    output.mkdir(parents=True, exist_ok=True)
    destination = output / 'comparison.csv'
    with destination.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    render_training([str(d) for d in directories], output / 'comparison.png', settings['x'], settings['smooth'])
    for row in rows:
        print(row)
    print('Comparison:', destination)
