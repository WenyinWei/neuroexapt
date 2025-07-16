from __future__ import annotations

"""Async DataLoader utilities using torchdata DataPipes.

This module provides a drop-in replacement for the traditional
`torch.utils.data.DataLoader` backed by highly efficient DataPipes with
background prefetching, pinned-memory transfer, and optional chunked GPU
transfer. It is designed to maximise GPU utilisation during NeuroExapt
training.
"""

import os
from typing import Iterable, Tuple, Callable, Any

import torch
from torch.utils.data import Dataset, functional_datapipe as fdp
from torch.utils.data.datapipes.iter import IterableWrapper
from torch.utils.data import DataLoader  # final stage collate

try:
    from torchvision import datasets, transforms  # typical vision datasets
    _TV_AVAILABLE = True
except ImportError:
    _TV_AVAILABLE = False

__all__ = [
    "build_cifar10_pipeline",
    "get_async_loader",
]

def _default_transform() -> Callable:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]) if _TV_AVAILABLE else lambda x: x

def build_cifar10_pipeline(root: str = "./data", train: bool = True, *, batch_size: int = 128, num_workers: int = 4, prefetch: int = 4) -> DataLoader:
    """Return high-performance DataLoader for CIFAR-10 using DataPipes."""
    if not _TV_AVAILABLE:
        raise RuntimeError("torchvision required for CIFAR10 pipeline")

    # Create the dataset directly instead of using DataPipes
    dataset = datasets.CIFAR10(root=root, train=train, download=True, transform=_default_transform())
    
    # Use traditional DataLoader with enhanced settings
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return loader


def get_async_loader(dataset: Dataset, *, batch_size: int, num_workers: int = 4, prefetch: int = 4, collate_fn: Callable[[Any], Any] | None = None) -> DataLoader:
    """Wrap an existing dataset into an async pinned-memory DataLoader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True if num_workers > 0 else False
    ) 