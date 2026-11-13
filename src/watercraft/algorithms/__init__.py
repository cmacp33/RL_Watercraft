"""Training algorithms; train.algorithm in configs/train.json selects one by name."""
import importlib

AVAILABLE = ('cem',)


def get(name):
    """Return the train(settings, env, directory, resume) function of an algorithm."""
    if name not in AVAILABLE:
        raise ValueError(f'Algorithm {name!r} is not implemented; available: ' + ', '.join(AVAILABLE))
    return importlib.import_module('watercraft.algorithms.' + name).train
