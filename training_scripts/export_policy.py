"""Export the checkpoint selected in configs/evaluate.json; takes no arguments."""
from watercraft.policy import export
from watercraft.config import load, run_path

def main():
    config = load('evaluate')
    directory = run_path(config['run'])
    filename = directory / 'checkpoints' / config['checkpoint']
    if not filename.exists():
        filename = directory / config['checkpoint']
    if filename.suffix == '.npz':
        raise ValueError('Select the source Torch checkpoint in configs/evaluate.json to export it')
    print(export(filename))

if __name__ == '__main__':
    main()
