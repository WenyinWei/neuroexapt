"""
Models module for NeuroExapt framework.
"""

# Import enhanced model components
from .enhanced_model_factory import (
    create_enhanced_model,
    get_model_stats,
    EnhancedTrainingConfig,
    get_enhanced_transforms,
    mixup_data,
    mixup_criterion,
    LabelSmoothingCrossEntropy,
    EnhancedResNet,
    HybridResNetDense,
    SEBlock,
    EnhancedBasicBlock,
    EnhancedBottleneck,
    DenseBlock
)

# Import existing enhanced resnet if available
try:
    from .enhanced_resnet import *
except ImportError:
    pass

__all__ = [
    # Enhanced model factory
    'create_enhanced_model',
    'get_model_stats',
    'EnhancedTrainingConfig',
    'get_enhanced_transforms',
    'mixup_data',
    'mixup_criterion',
    'LabelSmoothingCrossEntropy',
    
    # Model architectures
    'EnhancedResNet',
    'HybridResNetDense',
    'SEBlock',
    'EnhancedBasicBlock',
    'EnhancedBottleneck',
    'DenseBlock'
]