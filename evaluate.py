"""Edit configs/evaluate.json, then run python evaluate.py."""
import copy
from pathlib import Path
import numpy as np
import json
from watercraft.config import resolve, run_path, load
from watercraft.logging import CsvLog, evaluation_files, checkpoint_hash
from watercraft.envs import make, close, stop
from watercraft.policy import load as load_policy

EVAL_FIELDS = ['episode', 'total_reward', 'steps', 'start_dist_m', 'final_dist_m',
               'min_dist_m', 'reached', 'goal_x_m', 'goal_y_m', 'end_reason']


def evaluate(settings, env, actor, log):
    options = settings['evaluate']
    pool = env.unwrapped
    rows = []
    for episode in range(options['episodes']):
        observation = env.reset()
        start = minimum = pool.goal_distance_m()
        total, steps, done = 0.0, 0, False
        try:
            while not done:
                action = actor.act(observation)
                if options['noise']:
                    action = np.clip(action + np.random.normal(0, options['noise'], size=2), -1, 1)
                observation, reward, done, info = env.step(action)
                total += reward
                steps += 1
                minimum = min(minimum, info['goal_dist_m'])
            final = pool.goal_distance_m()
            reached = final <= getattr(pool, 'goal_radius_m', settings['task']['goal_radius_m'])
            row = dict(episode=episode, total_reward=total, steps=steps, start_dist_m=start,
                final_dist_m=final, min_dist_m=minimum, reached=int(reached),
                goal_x_m=float(pool.goal[0]), goal_y_m=float(pool.goal[1]),
                end_reason='goal' if reached else 'timeout')
            rows.append(row)
            log.write(**row)
            print(f'Episode {episode}: distance={final:.3f}m, reached={reached}, steps={steps}')
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
        if options['environment'] == 'real' and saved['train']['environment'] == 'real':
            # Preserve trained task settings while using current device connection/arming settings.
            for key in ('goal', 'goal_radius_m', 'max_episode_steps'):
                settings['real'][key] = saved['real'][key]
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
        settings['real']['goal'] = options['goal']
    actor = load_policy(checkpoint, settings['policy'])
    csv_path, json_path = evaluation_files(directory, options['environment'])
    metadata = dict(settings, checkpoint=str(checkpoint.relative_to(directory)), checkpoint_sha256=checkpoint_hash(checkpoint), status='started')
    json_path.write_text(json.dumps(metadata, indent=2))
    env = None
    np.random.seed(options['seed'])
    log = CsvLog(csv_path, EVAL_FIELDS)
    try:
        env = make(settings, 'evaluate', options['seed'])
        evaluate(settings, env, actor, log)
        metadata['status'] = 'complete'
    except BaseException as error:
        metadata.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        log.close()
        try:
            if env is not None:
                close(env)
        finally:
            json_path.write_text(json.dumps(metadata, indent=2))
    if options['auto_plot']:
        from watercraft.results import plot_evaluation
        plot_evaluation(csv_path)
    print('Evaluation:', csv_path)


if __name__ == '__main__':
    main()
