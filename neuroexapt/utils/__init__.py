"""
Utility modules for Neuro Exapt.
"""

# Core utilities (always available)
from .logging import setup_logger, log_metrics
from .gpu_manager import get_device, ensure_cuda_device, gpu_manager, GPUManager

# Optional utilities with graceful fallbacks
_OPTIONAL_COMPONENTS = []

try:
    from .dataset_loader import AdvancedDatasetLoader
    _OPTIONAL_COMPONENTS.append('AdvancedDatasetLoader')
except ImportError:
    # Create a placeholder for missing AdvancedDatasetLoader
    class AdvancedDatasetLoader:
        def __init__(self, *args, **kwargs):
            raise ImportError("AdvancedDatasetLoader requires additional dependencies (requests, etc.)")

try:
    from .visualization import ModelVisualizer, beautify_complexity
    _OPTIONAL_COMPONENTS.extend(['ModelVisualizer', 'beautify_complexity'])
except ImportError:
    # Create placeholders for missing visualization components
    class ModelVisualizer:
        def __init__(self, *args, **kwargs):
            raise ImportError("ModelVisualizer requires matplotlib and other visualization dependencies")
    
    def beautify_complexity(*args, **kwargs):
        raise ImportError("beautify_complexity requires matplotlib and other visualization dependencies")

__all__ = [
    # Core utilities
    'setup_logger', 'log_metrics',
    'get_device', 'ensure_cuda_device', 'gpu_manager', 'GPUManager',
    # Optional utilities (may raise ImportError if dependencies missing)
    'AdvancedDatasetLoader', 'ModelVisualizer', 'beautify_complexity'
] 