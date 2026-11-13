"""The existing MLP and both Torch/NumPy checkpoint formats."""
from pathlib import Path
import numpy as np
from watercraft.contract import NumpyPolicy, normalized_action


def network(observation_size, hidden_size, action_size):
    return __getattr__("Net")(observation_size, hidden_size, action_size)


# Torch is only imported when training or loading Torch policies.
def __getattr__(name):
    if name != 'Net':
        raise AttributeError(name)
    if "Net" in globals():
        return globals()["Net"]
    import torch.nn as nn
    class Net(nn.Module):
        def __init__(self, obs_size, hidden_size, n_actions):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(obs_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, n_actions))
        def forward(self, x):
            return self.net(x)
    globals()['Net'] = Net
    return Net


def build(config):
    return __getattr__('Net')(config['observation_size'], config['hidden_size'], config['action_size'])


def torch_state(filename):
    import torch
    if Path(filename).suffix == '.npz':
        with np.load(filename) as weights:
            return {key: torch.as_tensor(weights[array].copy()) for key, array in
                    [('net.0.weight', 'w0'), ('net.0.bias', 'b0'), ('net.2.weight', 'w1'), ('net.2.bias', 'b1')]}
    payload = torch.load(filename, map_location='cpu', weights_only=False)
    return payload.get('model', payload)


def load(filename, config):
    filename = Path(filename)
    if filename.suffix == '.npz':
        return NumpyPolicy(filename)
    model = build(config)
    model.load_state_dict(torch_state(filename))
    model.eval()
    return TorchPolicy(model)


class TorchPolicy:
    def __init__(self, model):
        self.model = model
    def act(self, observation):
        import torch
        with torch.no_grad():
            return normalized_action(torch.tanh(self.model(torch.as_tensor(np.asarray(observation), dtype=torch.float32))).cpu().numpy())


def export(pth_path, npz_path=None):
    state = torch_state(pth_path)
    output = Path(npz_path) if npz_path else Path(pth_path).with_suffix('.npz')
    np.savez(output, w0=state['net.0.weight'].numpy(), b0=state['net.0.bias'].numpy(),
             w1=state['net.2.weight'].numpy(), b1=state['net.2.bias'].numpy())
    return str(output)
