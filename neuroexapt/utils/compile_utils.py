"""Submodule-level torch.compile helpers.

`torch.compile` on a monolithic model may take long graph capture time。
本工具允许仅针对满足条件的子模块（如 Cell/MixedOp）执行编译，
可在保持调试友好的同时获取局部 fusing 与 kernel 优化收益。
"""
from __future__ import annotations

from typing import Callable, Any, Type
import torch
from torch import nn

__all__ = ["compile_submodules"]

def compile_submodules(model: nn.Module, *, predicate: Callable[[nn.Module], bool], backend: str = "inductor", fullgraph: bool = False):
    """Compile in-place all submodules satisfying predicate.

    Example::
        from neuroexapt.utils.compile_utils import compile_submodules
        compile_submodules(net, predicate=lambda m: m.__class__.__name__=="Cell")
    """
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is not available in this PyTorch build")

    for name, module in model.named_modules():
        if module is model:
            continue
        if predicate(module):
            compiled = torch.compile(module, backend=backend, fullgraph=fullgraph)
            parent_path = name.split(".")[:-1]
            attr_name = name.split(".")[-1]
            parent = model
            for p in parent_path:
                parent = getattr(parent, p)
            setattr(parent, attr_name, compiled) 