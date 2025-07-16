"""Module fusion utilities for NeuroExapt.

This helper performs operator fusion (Conv+BN+(ReLU)) using
`torch.ao.quantization.fuse_modules` to reduce kernel launches during
inference. It can be invoked once the model is finalised for deployment.
"""
from __future__ import annotations

from typing import List

import torch
from torch import nn

__all__ = ["fuse_model"]

def _fuse_sequential(sequential: nn.Sequential):
    """Fuse supported patterns in a Sequential module in-place."""
    modules = list(sequential._modules.keys())
    # Search for Conv-BN or Conv-BN-ReLU patterns
    idx = 0
    while idx < len(modules) - 1:
        # pattern length 3
        if idx + 2 < len(modules):
            names3 = modules[idx:idx+3]
            m1, m2, m3 = [sequential._modules[n] for n in names3]
            if isinstance(m1, nn.Conv2d) and isinstance(m2, nn.BatchNorm2d) and isinstance(m3, nn.ReLU):
                torch.ao.quantization.fuse_modules(sequential, names3, inplace=True)
                idx += 3
                continue
        # pattern length 2
        names2 = modules[idx:idx+2]
        m1, m2 = [sequential._modules[n] for n in names2]
        if isinstance(m1, nn.Conv2d) and isinstance(m2, nn.BatchNorm2d):
            torch.ao.quantization.fuse_modules(sequential, names2, inplace=True)
            idx += 2
            continue
        idx += 1


def fuse_model(model: nn.Module) -> nn.Module:
    """Fuse Conv+BN(+ReLU) for entire model. Operates in-place and returns model."""
    for name, module in model.named_children():
        # Recurse first
        fuse_model(module)
        # If this child is Sequential, attempt fusion inside
        if isinstance(module, nn.Sequential):
            _fuse_sequential(module)
        # Specific known blocks
        if isinstance(module, nn.Conv2d):
            # Look ahead for bn+relu controlled via parent sequential; skip here
            pass
    return model 