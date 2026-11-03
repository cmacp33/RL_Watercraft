#!/usr/bin/env python3
"""Export policy weights (.pth, needs torch) to .npz for policy_contract.NumpyPolicy (numpy only).

Usage:
    python3 training_scripts/export_policy.py runs/<run>/best.pth            # -> runs/<run>/best.npz
    python3 training_scripts/export_policy.py saved_weights/pool_policy.pth out.npz
"""
import os
import sys

import numpy as np
import torch


def export(pth_path, npz_path=None):
    npz_path = npz_path or os.path.splitext(pth_path)[0] + '.npz'
    state = torch.load(pth_path, map_location='cpu')
    np.savez(npz_path,
             w0=state['net.0.weight'].numpy(), b0=state['net.0.bias'].numpy(),
             w1=state['net.2.weight'].numpy(), b1=state['net.2.bias'].numpy())
    return npz_path


if __name__ == '__main__':
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    print("Wrote", export(*sys.argv[1:]))
