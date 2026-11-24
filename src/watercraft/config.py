"""Load settings relative to this checkout, independent of the shell directory."""
from pathlib import Path
import copy
import json

ROOT = Path(__file__).resolve().parents[2]

# One file per command. Each name is a section of that file, or the whole file.
SECTIONS = {'train': 'configs/train.json', 'algorithm': 'configs/train.json', 'task': 'configs/train.json',
            'sim': 'configs/train.json', 'policy': 'configs/train.json',
            'evaluate': 'configs/evaluate.json', 'results': 'configs/evaluate.json'}
FILES = {'deploy': 'configs/deploy.json', 'vision_tracking': 'vision_tracking/config.json'}


def path(value):
    p = Path(value).expanduser()
    return p if p.is_absolute() else ROOT / p


def load(name):
    filename = SECTIONS.get(name) or FILES[name]
    with (ROOT / filename).open() as stream:
        settings = json.load(stream)
    if name in SECTIONS:
        settings = settings.get(name) if isinstance(settings, dict) else None
    if not isinstance(settings, dict):
        raise ValueError(f'{filename} must contain a mapping for {name}')
    return settings


def run_path(name):
    if not isinstance(name, str) or not name or Path(name).name != name or name in ('.', '..'):
        raise ValueError('run must be a single folder name under runs/')
    return ROOT / 'runs' / name


def resolve(operation):
    settings = {key: load(key) for key in ('policy', 'task', 'sim')}
    settings[operation] = load(operation)
    run_path(settings[operation]['run'])
    if operation == 'train':
        from watercraft.algorithms import AVAILABLE
        algorithm = settings[operation]['algorithm']
        if algorithm not in AVAILABLE:
            raise ValueError(f'Algorithm {algorithm!r} is not implemented; available: ' + ', '.join(AVAILABLE))
        settings['algorithm'] = load('algorithm')
    validate(settings, operation)
    return copy.deepcopy(settings)


def validate(settings, operation):
    import math
    def positive(group, key):
        value = settings[group][key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{group}.{key} must be positive and finite')
    for group, keys in {'policy': ('hidden_size', 'position_scale_m', 'forward_speed_scale_mps', 'yaw_rate_scale_radps'),
                        'task': ('max_episode_steps', 'goal_radius_m'),
                        'sim': ('control_period_s',)}.items():
        for key in keys:
            positive(group, key)
    def integer(group, key):
        positive(group, key)
        if not isinstance(settings[group][key], int):
            raise ValueError(f'{group}.{key} must be an integer')
    integer('policy', 'hidden_size')
    integer('task', 'max_episode_steps')
    if 'goal_hold_steps' in settings['task']:
        integer('task', 'goal_hold_steps')
        if settings['task']['goal_hold_steps'] >= settings['task']['max_episode_steps']:
            raise ValueError('task.goal_hold_steps must be smaller than task.max_episode_steps')
    for group in ('task',):
        goal = settings[group]['goal']
        if not isinstance(goal, list) or len(goal) != 2 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in goal):
            raise ValueError(f'{group}.goal must contain two finite coordinates')
    if settings['policy']['observation_size'] != 6 or settings['policy']['action_size'] != 2:
        raise ValueError('The preserved policy contract requires six observations and two actions')
    if operation == 'train':
        # The algorithm section holds only the settings of the algorithm selected in train.algorithm.
        if settings['train']['algorithm'] == 'ppo':
            for key in ('steps_per_iteration', 'learning_rate', 'clip_range', 'epochs', 'minibatch_size',
                        'max_grad_norm', 'reward_scale', 'stop_iterations'):
                positive('algorithm', key)
            for key in ('steps_per_iteration', 'epochs', 'minibatch_size', 'stop_iterations'):
                integer('algorithm', key)
            for key in ('gamma', 'gae_lambda'):
                if not 0 < settings['algorithm'][key] <= 1:
                    raise ValueError(f'algorithm.{key} must be in (0, 1]')
            for key in ('value_coef', 'entropy_coef'):
                if settings['algorithm'][key] < 0:
                    raise ValueError(f'algorithm.{key} must not be negative')
        elif settings['train']['algorithm'] == 'sac':
            for key in ('learning_rate', 'tau', 'batch_size', 'buffer_size', 'updates_per_step', 'critic_hidden',
                        'initial_alpha', 'reward_scale', 'episodes_per_iteration', 'eval_every', 'eval_episodes',
                        'stop_iterations'):
                positive('algorithm', key)
            for key in ('batch_size', 'buffer_size', 'updates_per_step', 'critic_hidden', 'episodes_per_iteration',
                        'eval_every', 'eval_episodes', 'stop_iterations'):
                integer('algorithm', key)
            start = settings['algorithm']['start_steps']
            if not isinstance(start, int) or isinstance(start, bool) or start < 0:
                raise ValueError('algorithm.start_steps must be a nonnegative integer')
            if not 0 < settings['algorithm']['gamma'] <= 1 or not 0 < settings['algorithm']['tau'] <= 1:
                raise ValueError('algorithm.gamma and algorithm.tau must be in (0, 1]')
        else:
            for key in ('batch_size', 'learning_rate', 'noise_decay_iterations', 'elite_memory', 'train_steps_per_iteration', 'reference_speed_mps', 'stop_iterations'):
                positive('algorithm', key)
            for key in ('batch_size', 'elite_memory', 'train_steps_per_iteration', 'stop_iterations'):
                integer('algorithm', key)
            if not 0 <= settings['algorithm']['percentile'] <= 100:
                raise ValueError('algorithm.percentile must be in [0, 100]')
        positive('train', 'minutes')
        limit = settings['train']['max_iterations']
        if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 1):
            raise ValueError('max_iterations must be null or a positive integer')
        if settings['train']['resume'] and settings['train']['initialize_from']:
            raise ValueError('Choose resume or initialize_from, not both')
    else:
        integer('evaluate', 'episodes')
        noise = settings['evaluate']['noise']
        if not isinstance(noise, (int, float)) or not math.isfinite(noise) or noise < 0:
            raise ValueError('evaluate.noise must be finite and nonnegative')
        if settings['evaluate']['goal'] is not None and settings['evaluate']['random_goals']:
            raise ValueError('Choose a fixed goal or random goals, not both')
        limit = settings['evaluate'].get('max_episode_steps')
        if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 1):
            raise ValueError('evaluate.max_episode_steps must be null or a positive integer')
        hold = settings['evaluate'].get('goal_hold_steps')
        if hold is not None and (not isinstance(hold, int) or isinstance(hold, bool) or hold < 1):
            raise ValueError('evaluate.goal_hold_steps must be null or a positive integer')
