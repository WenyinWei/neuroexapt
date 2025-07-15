"""CUDA accelerated operations for NeuroExapt."""

import torch

from .softmax_sum import SoftmaxSumFunction

__all__ = ["SoftmaxSumFunction", "CUDA_AVAILABLE"]

CUDA_AVAILABLE = torch.cuda.is_available() 