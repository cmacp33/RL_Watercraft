"""Load settings relative to this checkout, independent of the shell directory."""
from pathlib import Path
import copy
import json

ROOT = Path(__file__).resolve().parents[1]


def path(value):
    p = Path(value).expanduser()
    return p if p.is_absolute() else ROOT / p


def load(name):
    with (ROOT / 'configs' / (name + '.json')).open() as stream:
        settings = json.load(stream)
    if not isinstance(settings, dict):
        raise ValueError(f'configs/{name}.json must contain a mapping')
    return settings


def run_path(name):
    if not isinstance(name, str) or not name or Path(name).name != name or name in ('.', '..'):
        raise ValueError('run must be a single folder name under runs/')
    return ROOT / 'runs' / name


def resolve(operation):
    settings = {key: load(key) for key in ('policy', 'task', 'sim', 'real')}
    settings[operation] = load(operation)
    env = settings[operation]['environment']
    if env not in ('sim', 'real'):
        raise ValueError('environment must be sim or real')
    run_path(settings[operation]['run'])
    if operation == 'train':
        algorithm = settings[operation]['algorithm']
        if algorithm != 'cem':
            raise ValueError(f'Algorithm {algorithm!r} is not implemented; available: cem')
        settings['algorithm'] = load('algorithms/' + algorithm)
    if env == 'real':
        for key in ('goal', 'goal_radius_m', 'max_episode_steps'):
            settings['task'][key] = copy.deepcopy(settings['real'][key])
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
                        'sim': ('control_period_s',),
                        'real': ('control_period_s', 'goal_radius_m', 'max_episode_steps', 'measurement_timeout_s')}.items():
        for key in keys:
            positive(group, key)
    def integer(group, key):
        positive(group, key)
        if not isinstance(settings[group][key], int):
            raise ValueError(f'{group}.{key} must be an integer')
    integer('policy', 'hidden_size')
    integer('task', 'max_episode_steps')
    integer('real', 'max_episode_steps')
    for group in ('task', 'real'):
        goal = settings[group]['goal']
        if not isinstance(goal, list) or len(goal) != 2 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in goal):
            raise ValueError(f'{group}.goal must contain two finite coordinates')
    if not isinstance(settings['real']['arm'], bool):
        raise ValueError('real.arm must be true or false')
    if settings['policy']['observation_size'] != 6 or settings['policy']['action_size'] != 2:
        raise ValueError('The preserved policy contract requires six observations and two actions')
    if not 0 < settings['real']['max_action'] <= 1:
        raise ValueError('real.max_action must be in (0, 1]')
    if operation == 'train':
        for key in ('batch_size', 'learning_rate', 'noise_decay_iterations', 'elite_memory', 'train_steps_per_iteration', 'reference_speed_mps', 'stop_iterations'):
            positive('algorithm', key)
        for key in ('batch_size', 'elite_memory', 'train_steps_per_iteration', 'stop_iterations'):
            integer('algorithm', key)
        positive('train', 'minutes')
        if not 0 <= settings['algorithm']['percentile'] <= 100:
            raise ValueError('algorithm.percentile must be in [0, 100]')
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
