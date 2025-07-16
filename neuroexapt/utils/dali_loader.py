"""NVIDIA DALI accelerated dataloaders for NeuroExapt.

This module is optional – it requires `nvidia-dali` (GPU build) >=1.30.
It provides high-throughput pipelines that decode/augment images on GPU,
exposing PyTorch-compatible iterators.
"""
from __future__ import annotations

import os
from typing import Tuple, Optional

try:
    import nvidia.dali as dali  # type: ignore
    import nvidia.dali.fn as fn  # type: ignore
    import nvidia.dali.types as types  # type: ignore
    from nvidia.dali.plugin.pytorch import DALIGenericIterator  # type: ignore
    _DALI_AVAILABLE = True
except ImportError:  # pragma: no cover
    _DALI_AVAILABLE = False

__all__ = ["build_cifar10_dali", "is_dali_available"]

def is_dali_available() -> bool:
    return _DALI_AVAILABLE


def build_cifar10_dali(data_root: str = "./data", batch_size: int = 128, training: bool = True, num_threads: int = 4, device_id: int = 0, prefetch_queue: int = 2):
    """Return DALIGenericIterator for CIFAR-10.

    Parameters
    ----------
    data_root: str
        Path where CIFAR-10 tar files are/will be downloaded.
    training: bool
        Whether to build training pipeline (random flip/crop) or validation.
    """
    if not _DALI_AVAILABLE:
        raise RuntimeError("nvidia-dali-gpu is not installed. Please `pip install nvidia-dali-cuda11` (or matching CUDA) to use this feature.")

    cifar_dir = os.path.join(data_root, "cifar-10-batches-bin")
    if not os.path.exists(cifar_dir):
        # fallback to torchvision download to prepare binary files
        from torchvision.datasets import CIFAR10
        CIFAR10(root=data_root, train=True, download=True)

    pipeline = dali.pipeline.Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id, prefetch_queue_depth=prefetch_queue)

    with pipeline:
        # Read CIFAR-10 encoded examples (DALI has ready reader)
        inputs, labels = fn.readers.cifar10(path=data_root, random_shuffle=training)

        # Decode to GPU tensor (already uint8 HWC)
        images = inputs.gpu()

        # Cast & augment
        images = fn.crop_mirror_normalize(
            images,
            dtype=types.FLOAT,
            output_layout="CHW",
            crop=(32, 32),
            mean=[0.4914*255, 0.4822*255, 0.4465*255],
            std=[0.2023*255, 0.1994*255, 0.2010*255],
            mirror=fn.random.coin_flip(probability=0.5) if training else 0,
        )

        pipeline.set_outputs(images, labels)

    pipeline.build()

    # PyTorch iterator
    return DALIGenericIterator(pipeline, ["data", "label"], reader_name="Reader", last_batch_policy="PARTIAL") 