"""
Utility modules for Neuro Exapt.
"""

# Always available utilities
_BASE_UTILS = []

# Optional imports with fallbacks
_OPTIONAL_UTILS = []

try:
    from .dataset_loader import AdvancedDatasetLoader
    _OPTIONAL_UTILS.extend(['AdvancedDatasetLoader'])
except ImportError as e:
    print(f"Warning: Could not import dataset_loader: {e}")

try:
    from .visualization import ModelVisualizer, beautify_complexity
    _OPTIONAL_UTILS.extend(['ModelVisualizer', 'beautify_complexity'])
except ImportError as e:
    print(f"Warning: Could not import visualization: {e}")

try:
    from .logging import setup_logger, log_metrics
    _OPTIONAL_UTILS.extend(['setup_logger', 'log_metrics'])
except ImportError as e:
    print(f"Warning: Could not import logging: {e}")

try:
    from .gpu_manager import get_device, ensure_cuda_device, gpu_manager, GPUManager
    _OPTIONAL_UTILS.extend(['get_device', 'ensure_cuda_device', 'gpu_manager', 'GPUManager'])
except ImportError as e:
    print(f"Warning: Could not import gpu_manager: {e}")

__all__ = _BASE_UTILS + _OPTIONAL_UTILS 