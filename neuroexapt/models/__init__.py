"""
NeuroExapt Models Package
Enhanced model architectures for intelligent evolution
"""

from .enhanced_resnet import (
    EnhancedResNet, enhanced_resnet18, enhanced_resnet34, enhanced_resnet50,
    enhanced_wide_resnet, create_enhanced_model, EnhancedTrainingConfig,
    get_enhanced_transforms, LabelSmoothingCrossEntropy, mixup_data, mixup_criterion
)

__all__ = [
    'EnhancedResNet', 'enhanced_resnet18', 'enhanced_resnet34', 'enhanced_resnet50',
    'enhanced_wide_resnet', 'create_enhanced_model', 'EnhancedTrainingConfig',
    'get_enhanced_transforms', 'LabelSmoothingCrossEntropy', 'mixup_data', 'mixup_criterion'
]