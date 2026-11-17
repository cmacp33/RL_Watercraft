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


def _max_steps(filename):
    """Episode step limit, from the evaluation's metadata file (falls back to the longest episode)."""
    metadata = Path(filename).with_suffix('.json')
    if metadata.exists():
        value = json.loads(metadata.read_text()).get('task', {}).get('max_episode_steps')
        if isinstance(value, int) and value > 0:
            return value
    return None


def distance_curves(filename, max_steps=None):
    """Distance to goal per step for every episode, as a (episodes, max_steps + 1) array.

    Episodes that finish early (goal reached) keep their last distance until max_steps, so every
    step's median is taken over all episodes. Returns None when there is no per-step log, or when
    the goal differs between episodes (random goals: raw distances are not comparable).
    """
    from watercraft.evaluate import steps_path
    episodes = read_csv(filename)
    steps = read_csv(steps_path(filename))
    if not episodes or not steps:
        return None
    goals = {(r['goal_x_m'], r['goal_y_m']) for r in episodes}
    if len(goals) > 1:
        print('Distance curve skipped: goals differ between episodes (use a fixed goal)')
        return None
    by_episode = {}
    for row in steps:
        by_episode.setdefault(int(row['episode']), []).append((int(row['step']), float(row['dist_m'])))
    longest = max(step for values in by_episode.values() for step, _ in values)
    max_steps = max(max_steps or _max_steps(filename) or 0, longest)
    curves = np.empty((len(by_episode), max_steps + 1))
    for i, episode in enumerate(sorted(by_episode)):
        values = sorted(by_episode[episode])
        distances = np.array([d for _, d in values])
        curves[i, :len(distances)] = distances
        curves[i, len(distances):] = distances[-1]  # hold the last distance
    return curves


def distance_summary(filename, max_steps=None):
    """Single numbers for comparing evaluations with the same fixed goal."""
    curves = distance_curves(filename, max_steps)
    if curves is None:
        return None
    median = np.median(curves, axis=0)
    rows = read_csv(filename)
    reached = [int(r['steps']) for r in rows if r['reached'] == '1']
    return {
        'episodes': int(curves.shape[0]),
        'max_steps': int(curves.shape[1] - 1),
        'start_dist_m': round(float(median[0]), 4),
        'mean_median_dist_m': round(float(median.mean()), 4),  # area under the median curve / steps; lower is better
        'median_final_dist_m': round(float(median[-1]), 4),
        'success_rate': round(len(reached) / len(rows), 3),
        'median_steps_to_goal': float(np.median(reached)) if reached else None,
        # mean |change in model output| per step, averaged over both thrusters; 0 = constant, 2 = full flip every step
        'median_action_smoothness': (round(float(np.median([float(r['action_smoothness']) for r in rows])), 4)
                                     if rows and 'action_smoothness' in rows[0] else None),
    }


def plot_distance(filename, max_steps=None):
    """<timestamp>_distance.png: median distance to goal vs step, with the 25-75% band."""
    curves = distance_curves(filename, max_steps)
    if curves is None:
        return
    steps = np.arange(curves.shape[1])
    median = np.median(curves, axis=0)
    low, high = np.percentile(curves, [25, 75], axis=0)
    figure, axis = plt.subplots(figsize=(9, 5))
    for curve in curves:
        axis.plot(steps, curve, color='grey', alpha=0.2, linewidth=0.8)
    axis.fill_between(steps, low, high, color='C0', alpha=0.25, label='25-75% of episodes')
    axis.plot(steps, median, color='C0', linewidth=2.2, label=f'median ({curves.shape[0]} episodes)')
    metadata = Path(filename).with_suffix('.json')
    radius = json.loads(metadata.read_text()).get('task', {}).get('goal_radius_m') if metadata.exists() else None
    if radius:
        axis.axhline(radius, color='C2', linestyle='--', linewidth=1.2, label=f'goal radius {radius} m')
    axis.set(xlabel='Step', ylabel='Distance to goal (m)', xlim=(0, steps[-1]), ylim=(0, None),
             title='Distance to goal per step (finished episodes hold their last distance)')
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(Path(filename).with_name(Path(filename).stem + '_distance.png'), dpi=120)
    plt.close(figure)


def plot_episode_steps(axis, rows, max_steps=None):
    """Steps taken in each episode: green reached the goal, red timed out; dashed line = step limit."""
    episodes = [int(r['episode']) for r in rows]
    steps = [int(r['steps']) for r in rows]
    colors = ['tab:green' if int(r['reached']) else 'tab:red' for r in rows]
    from matplotlib.patches import Patch
    axis.bar(episodes, steps, color=colors)
    handles = [Patch(color='tab:green', label='reached'), Patch(color='tab:red', label='timed out'),
               axis.axhline(np.mean(steps), color='black', linewidth=1, label=f'mean {np.mean(steps):.0f}')]
    if max_steps:
        handles.append(axis.axhline(max_steps, color='gray', linestyle='--', linewidth=1, label=f'limit {max_steps}'))
    # headroom above the tallest bar / limit line so the legend never covers them
    axis.set(xlabel='Episode', ylabel='Steps in episode', ylim=(0, 1.3 * max(steps + [max_steps or 0])))
    axis.legend(handles=handles, loc='upper right', fontsize='small', ncol=2)


def plot_evaluation(filename):
    if str(filename).endswith('_steps.csv'):
        return
    rows = read_csv(filename)
    if not rows:
        return
    plot_distance(filename)
    smooth = 'action_smoothness' in rows[0]
    panels = 4 if smooth else 3
    figure, axes = plt.subplots(1, panels, figsize=(5 * panels, 4))
    episodes = [int(r['episode']) for r in rows]
    axes[0].plot(episodes, [float(r['final_dist_m']) for r in rows], marker='o')
    axes[0].set(xlabel='Episode', ylabel='Final distance (m)')
    axes[1].bar(episodes, [int(r['reached']) for r in rows])
    axes[1].set(xlabel='Episode', ylabel='Reached goal', ylim=(0, 1.1))
    plot_episode_steps(axes[2], rows, _max_steps(filename))
    if smooth:
        axes[3].bar(episodes, [float(r['action_smoothness']) for r in rows])
        axes[3].set(xlabel='Episode', ylabel='Mean |action change| per step (lower = smoother)', ylim=(0, None))
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
