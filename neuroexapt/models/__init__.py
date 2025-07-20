"""
Models module for NeuroExapt framework.
"""

# Import enhanced model components from core module
from ..core.enhanced_model_factory import (
    create_enhanced_model,
    EnhancedResNet,
    HybridResNetDense,
    EnhancedBasicBlock,
    EnhancedBottleneck,
    DenseBlock
)

# Import enhanced training components from local enhanced_resnet module
from .enhanced_resnet import (
    EnhancedTrainingConfig,
    get_enhanced_transforms,
    mixup_data,
    mixup_criterion,
    LabelSmoothingCrossEntropy,
    SEBlock
)

# Import existing enhanced resnet if available
try:
    from .enhanced_resnet import *
except ImportError:
    pass

__all__ = [
    # Enhanced model factory
    'create_enhanced_model',
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