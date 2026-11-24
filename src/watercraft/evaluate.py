"""Edit configs/evaluate.json, then run python -m watercraft.evaluate (or watercraft-evaluate)."""
import copy
from pathlib import Path
import numpy as np
import json
from watercraft.config import resolve, run_path, load
from watercraft.logging import CsvLog, evaluation_files, checkpoint_hash
from watercraft.sim_env import make, close, stop
from watercraft.policy import load as load_policy

EVAL_FIELDS = ['episode', 'total_reward', 'steps', 'start_dist_m', 'final_dist_m',
               'min_dist_m', 'reached', 'goal_x_m', 'goal_y_m', 'end_reason', 'action_smoothness']
# One row per control step (step 0 = right after reset, no action yet).
#   model_*: the policy's output, in [-1, 1].
#   motor_*: the command published to the thruster plugin (-1..1 = full reverse..full forward thrust), i.e. the model output
#            after evaluation noise and, with randomize on, the simulated motor delay, gain and deadband.
#   action_change: (|change in model_left| + |change in model_right|) / 2 since the previous step.
STEP_FIELDS = ['episode', 'step', 'dist_m', 'model_left', 'model_right', 'motor_left', 'motor_right',
               'action_change']


def steps_path(csv_path):
    """<timestamp>.csv -> <timestamp>_steps.csv, next to the per-episode file."""
    csv_path = Path(csv_path)
    return csv_path.with_name(csv_path.stem + '_steps.csv')


def evaluate(settings, env, actor, log, step_log=None):
    options = settings['evaluate']
    pool = env.unwrapped
    rows = []
    for episode in range(options['episodes']):
        observation = env.reset()
        start = minimum = pool.goal_distance_m()
        total, steps, done = 0.0, 0, False
        if step_log is not None:
            step_log.write(episode=episode, step=0, dist_m=float(start))
        previous, changes = None, []
        try:
            while not done:
                output = np.asarray(actor.act(observation), dtype=np.float64)
                action = output
                if options['noise']:
                    action = np.clip(output + np.random.normal(0, options['noise'], size=2), -1, 1)
                observation, reward, done, info = env.step(action)
                total += reward
                steps += 1
                minimum = min(minimum, info['goal_dist_m'])
                change = '' if previous is None else float(np.abs(output - previous).mean())
                if change != '':
                    changes.append(change)
                previous = output
                if step_log is not None:
                    motor = np.asarray(info.get('applied_action', action), dtype=np.float64)
                    step_log.write(episode=episode, step=steps, dist_m=float(info['goal_dist_m']),
                                   model_left=float(output[0]), model_right=float(output[1]),
                                   motor_left=float(motor[0]), motor_right=float(motor[1]),
                                   action_change=change)
            final = pool.goal_distance_m()
            # reached = inside the goal radius for task.goal_hold_steps steps in a row (1 = first touch)
            reached = bool(info['reached'])
            row = dict(episode=episode, total_reward=total, steps=steps, start_dist_m=start,
                final_dist_m=final, min_dist_m=minimum, reached=int(reached),
                goal_x_m=float(pool.goal[0]), goal_y_m=float(pool.goal[1]),
                end_reason='goal' if reached else 'timeout',
                action_smoothness=float(np.mean(changes)) if changes else 0.0)
            rows.append(row)
            log.write(**row)
            print(f'Episode {episode}: distance={final:.3f}m, reached={reached}, steps={steps}, '
                  f'action change/step={row["action_smoothness"]:.3f}')
        finally:
            stop(env)
    return rows


def main():
    settings = resolve('evaluate')
    options = settings['evaluate']
    directory = run_path(options['run'])
    if not directory.is_dir():
        raise FileNotFoundError(f'Run does not exist: {directory}')
    checkpoint_name = Path(options['checkpoint'])
    if checkpoint_name.is_absolute() or '..' in checkpoint_name.parts:
        raise ValueError('checkpoint must be relative to the run checkpoint folder')
    checkpoint = directory / 'checkpoints' / checkpoint_name
    if not checkpoint.exists():
        # Existing historical runs keep their flat layout.
        checkpoint = directory / checkpoint_name
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    saved_path = directory / 'config.json'
    saved = json.loads(saved_path.read_text()) if saved_path.exists() else {}
    if all(group in saved for group in ('policy', 'task', 'sim', 'train')):
        # Policy scales/architecture always come from training, not today's defaults.
        for group in ('policy', 'task', 'sim'):
            settings[group] = copy.deepcopy(saved[group])
    else:
        metadata = directory / 'config.json'
        if metadata.exists():
            old = json.loads(metadata.read_text())
            settings['policy']['hidden_size'] = old.get('hyperparams', {}).get('hidden_size', settings['policy']['hidden_size'])
            settings['policy']['position_scale_m'] = old.get('position_scale_m', settings['policy']['position_scale_m'])
            # Config-driven contract defaults must agree with checkpoint preprocessing.
    from watercraft import contract
    contract.FORWARD_SPEED_SCALE_MPS = settings['policy']['forward_speed_scale_mps']
    contract.YAW_RATE_SCALE_RADPS = settings['policy']['yaw_rate_scale_radps']
    settings['sim']['random_goals'] = options['random_goals']
    settings['sim']['domain_randomization'] = options['randomize']
    if options['goal'] is not None:
        settings['task']['goal'] = options['goal']
    # null keeps the step limit the run was trained with.
    if options.get('max_episode_steps') is not None:
        settings['task']['max_episode_steps'] = options['max_episode_steps']
    # null keeps the hold the run was trained with (1 = first touch, for runs saved before it existed).
    if options.get('goal_hold_steps') is not None:
        settings['task']['goal_hold_steps'] = options['goal_hold_steps']
    actor = load_policy(checkpoint, settings['policy'])
    csv_path, json_path = evaluation_files(directory, 'sim')
    metadata = dict(settings, checkpoint=str(checkpoint.relative_to(directory)), checkpoint_sha256=checkpoint_hash(checkpoint), status='started')
    json_path.write_text(json.dumps(metadata, indent=2))
    env = None
    np.random.seed(options['seed'])
    log = CsvLog(csv_path, EVAL_FIELDS)
    step_log = CsvLog(steps_path(csv_path), STEP_FIELDS)
    try:
        env = make(settings, options['seed'])
        evaluate(settings, env, actor, log, step_log)
        metadata['status'] = 'complete'
    except BaseException as error:
        metadata.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        log.close()
        step_log.close()
        try:
            if env is not None:
                close(env)
        finally:
            json_path.write_text(json.dumps(metadata, indent=2))
    from watercraft.results import distance_summary
    summary = distance_summary(csv_path, settings['task']['max_episode_steps'])
    if summary:
        metadata['distance_curve'] = summary
        json_path.write_text(json.dumps(metadata, indent=2))
        print('Distance curve:', ', '.join(f'{k}={v}' for k, v in summary.items()))
    if options['auto_plot']:
        from watercraft.results import plot_evaluation
        plot_evaluation(csv_path)
    print('Evaluation:', csv_path)


if __name__ == '__main__':
    main()
