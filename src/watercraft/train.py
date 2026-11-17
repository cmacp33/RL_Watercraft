"""Edit configs/train.json, then run python -m watercraft.train (or watercraft-train)."""
from watercraft.config import resolve, path, run_path


def main():
    import random
    import numpy as np
    import torch
    from watercraft.logging import create_run
    from watercraft.sim_env import make, close
    from watercraft.algorithms import get
    from watercraft.policy import torch_load
    settings = resolve('train')
    options = settings['train']
    train = get(options['algorithm'])
    resume = None
    if options['resume']:
        filename = path(options['resume'])
        resume = torch_load(filename)
        if not {'model', 'optimizer', 'settings', 'iteration'} <= resume.keys():
            raise ValueError('resume must reference a complete latest.pt training checkpoint')
        previous = resume['settings']
        for key in ('policy', 'task', 'sim', 'algorithm'):
            if settings[key] != previous[key]:
                raise ValueError(f'Resume configuration differs in {key}; restore it or start a new run')
        if any(options[k] != previous['train'][k] for k in ('algorithm', 'run', 'seed')):
            raise ValueError('Resume must use the original algorithm, run and seed')
        directory = filename.parent.parent
        if directory.resolve() != run_path(options['run']).resolve():
            raise ValueError('Resume checkpoint must belong to the configured run')
        latest = directory / 'checkpoints/latest.pt'
        if filename.resolve() != latest.resolve():
            raise ValueError('Resume the latest checkpoint; use initialize_from for older weights')
    else:
        if options['initialize_from'] and not path(options['initialize_from']).is_file():
            raise FileNotFoundError(path(options['initialize_from']))
        directory = run_path(options['run'])
        if directory.exists():
            raise FileExistsError(f'Run already exists: {directory}')
    random.seed(options['seed'])
    np.random.seed(options['seed'])
    torch.manual_seed(options['seed'])
    env = None
    try:
        env = make(settings, options['seed'])
        if not resume:
            directory = create_run(settings)
        train(settings, env, directory, resume)
    finally:
        if env is not None:
            close(env)
    if options['auto_plot']:
        from watercraft.results import plot_run
        plot_run(directory)
    print('Outputs:', directory)


if __name__ == '__main__':
    main()
