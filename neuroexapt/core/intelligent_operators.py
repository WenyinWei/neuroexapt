"""
Intelligent Operators for Smart Layer Selection and Data Flow Management.

This module implements advanced operators that can:
1. Intelligently select layer types based on information-theoretic metrics
2. Dynamically adjust data flow sizes based on feature complexity
3. Create specialized branches for different feature extraction needs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from abc import ABC, abstractmethod
import copy
from collections import OrderedDict
from .operators import StructuralOperator
from .information_theory import InformationBottleneck


class LayerTypeSelector:
    """Intelligent layer type selection based on information metrics."""
    
    def __init__(self, ib: Optional[InformationBottleneck] = None):
        self.ib = ib or InformationBottleneck()
        
        # Layer type characteristics
        self.layer_properties = {
            'conv': {
                'good_for': ['spatial_features', 'local_patterns', 'edge_detection'],
                'complexity': 'medium',
                'param_efficiency': 0.7
            },
            'pooling': {
                'good_for': ['dimension_reduction', 'invariance', 'noise_reduction'],
                'complexity': 'low',
                'param_efficiency': 1.0  # No parameters
            },
            'attention': {
                'good_for': ['long_range_dependencies', 'feature_selection', 'context'],
                'complexity': 'high',
                'param_efficiency': 0.4
            },
            'depthwise_conv': {
                'good_for': ['channel_wise_features', 'efficiency', 'mobile'],
                'complexity': 'low',
                'param_efficiency': 0.9
            },
            'bottleneck': {
                'good_for': ['dimension_reduction', 'feature_compression', 'efficiency'],
                'complexity': 'medium',
                'param_efficiency': 0.8
            }
        }
    
    def select_layer_type(
        self,
        input_tensor: torch.Tensor,
        layer_metrics: Dict[str, float],
        target_task: str = "classification"
    ) -> str:
        """
        Select optimal layer type based on input characteristics and metrics.
        
        Args:
            input_tensor: Input tensor to analyze
            layer_metrics: Current layer metrics (entropy, MI, etc.)
            target_task: Target task type
            
        Returns:
            Selected layer type
        """
        # Analyze input characteristics
        spatial_complexity = self._analyze_spatial_complexity(input_tensor)
        channel_redundancy = self._analyze_channel_redundancy(input_tensor)
        information_density = layer_metrics.get('mutual_information', 0.5)
        
        # Decision logic based on analysis
        if spatial_complexity > 0.7 and information_density > 0.6:
            # High spatial complexity and information - use attention
            return 'attention'
        elif channel_redundancy > 0.5:
            # High channel redundancy - use depthwise conv
            return 'depthwise_conv'
        elif layer_metrics.get('entropy', 1.0) < 0.3:
            # Low entropy - use pooling to reduce dimensions
            return 'pooling'
        elif information_density < 0.4 and spatial_complexity < 0.5:
            # Low information and spatial complexity - use bottleneck
            return 'bottleneck'
        else:
            # Default to standard convolution
            return 'conv'
    
    def _analyze_spatial_complexity(self, tensor: torch.Tensor) -> float:
        """Analyze spatial complexity of feature maps."""
        if tensor.dim() < 4:  # Not a spatial tensor
            return 0.5
        
        # Calculate gradient magnitude as proxy for spatial complexity
        if tensor.size(2) > 1 and tensor.size(3) > 1:
            dx = tensor[:, :, 1:, :] - tensor[:, :, :-1, :]
            dy = tensor[:, :, :, 1:] - tensor[:, :, :, :-1]
            gradient_magnitude = torch.sqrt(dx[:, :, :, :-1]**2 + dy[:, :, :-1, :]**2)
            complexity = gradient_magnitude.mean().item() / (tensor.std().item() + 1e-6)
            return min(1.0, complexity)
        return 0.5
    
    def _analyze_channel_redundancy(self, tensor: torch.Tensor) -> float:
        """Analyze redundancy across channels."""
        if tensor.dim() < 4 or tensor.size(1) < 2:
            return 0.0
        
        # Calculate correlation between channels
        channels = tensor.size(1)
        correlations = []
        
        for i in range(min(10, channels)):  # Sample first 10 channels
            for j in range(i+1, min(10, channels)):
                corr = F.cosine_similarity(
                    tensor[:, i].flatten().unsqueeze(0),
                    tensor[:, j].flatten().unsqueeze(0)
                ).item()
                correlations.append(abs(corr))
        
        return np.mean(correlations) if correlations else 0.0


class IntelligentExpansionOperator(StructuralOperator):
    """
    Intelligent network expansion with adaptive layer type selection
    and data flow management.
    """
    
    def __init__(
        self,
        layer_selector: Optional[LayerTypeSelector] = None,
        expansion_ratio: float = 0.1,
        min_feature_size: int = 7,
        device: Optional[torch.device] = None
    ):
        self.layer_selector = layer_selector or LayerTypeSelector()
        self.expansion_ratio = expansion_ratio
        self.min_feature_size = min_feature_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def apply(self, model: nn.Module, metrics: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
        """Apply intelligent expansion."""
        layer_importances = metrics.get('layer_importances', {})
        if not layer_importances:
            return model, {'message': 'No layer importance metrics available'}
        
        # Find expansion points
        expansion_points = self._find_expansion_points(layer_importances, model)
        if not expansion_points:
            return model, {'message': 'No suitable expansion points found'}
        
        # Create expanded model
        expanded_model = copy.deepcopy(model)
        expanded_layers = []
        
        for layer_name, layer_info in expansion_points:
            # Get layer activation for analysis
            activation = metrics.get(f'{layer_name}_activation')
            if activation is None:
                # Create dummy activation for analysis
                layer = dict(model.named_modules())[layer_name]
                activation = self._get_dummy_activation(layer)
            
            # Select optimal layer type
            layer_metrics = {
                'entropy': metrics.get(f'{layer_name}_entropy', 0.5),
                'mutual_information': layer_info['importance']
            }
            selected_type = self.layer_selector.select_layer_type(
                activation, layer_metrics, metrics.get('task_type', 'classification')
            )
            
            # Create and insert new layer
            new_layer = self._create_layer(selected_type, layer, activation)
            if new_layer:
                self._insert_layer_after(expanded_model, layer_name, new_layer, f"{layer_name}_{selected_type}")
                expanded_layers.append(f"{layer_name}_{selected_type}")
        
        return expanded_model, {
            'expanded_layers': expanded_layers,
            'layer_types_added': {name: name.split('_')[-1] for name in expanded_layers}
        }
    
    def can_apply(self, model: nn.Module, metrics: Dict[str, Any]) -> bool:
        """Check if intelligent expansion can be applied."""
        return bool(metrics.get('layer_importances'))
    
    def _find_expansion_points(
        self, 
        layer_importances: Dict[str, float], 
        model: nn.Module
    ) -> List[Tuple[str, Dict]]:
        """Find optimal points for expansion."""
        # Sort by importance
        sorted_layers = sorted(
            layer_importances.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Select top layers for expansion
        max_expand = int(len(sorted_layers) * self.expansion_ratio)
        expansion_points = []
        
        for layer_name, importance in sorted_layers[:max_expand]:
            # Check if layer exists and is expandable
            if self._is_expandable(model, layer_name):
                expansion_points.append((layer_name, {'importance': importance}))
        
        return expansion_points
    
    def _is_expandable(self, model: nn.Module, layer_name: str) -> bool:
        """Check if a layer can be expanded."""
        try:
            layer = dict(model.named_modules())[layer_name]
            return isinstance(layer, (nn.Conv2d, nn.Linear, nn.Conv1d))
        except:
            return False
    
    def _get_dummy_activation(self, layer: nn.Module) -> torch.Tensor:
        """Generate dummy activation for analysis."""
        if isinstance(layer, nn.Conv2d):
            return torch.randn(1, layer.out_channels, 32, 32, device=self.device)
        elif isinstance(layer, nn.Linear):
            return torch.randn(1, layer.out_features, device=self.device)
        else:
            return torch.randn(1, 64, 32, 32, device=self.device)
    
    def _create_layer(
        self, 
        layer_type: str, 
        reference_layer: nn.Module,
        activation: torch.Tensor
    ) -> Optional[nn.Module]:
        """Create new layer based on selected type."""
        if isinstance(reference_layer, nn.Conv2d):
            return self._create_conv_variant(layer_type, reference_layer, activation)
        elif isinstance(reference_layer, nn.Linear):
            return self._create_linear_variant(layer_type, reference_layer)
        return None
    
    def _create_conv_variant(
        self, 
        layer_type: str, 
        ref_layer: nn.Conv2d,
        activation: torch.Tensor
    ) -> nn.Module:
        """Create convolutional layer variant."""
        out_channels = ref_layer.out_channels
        
        if layer_type == 'pooling':
            # Adaptive pooling to reduce spatial dimensions
            current_size = activation.size(-1)
            if current_size > self.min_feature_size:
                target_size = max(self.min_feature_size, current_size // 2)
                return nn.AdaptiveAvgPool2d(target_size)
            else:
                return nn.Identity()  # Don't reduce further
                
        elif layer_type == 'attention':
            # Channel attention module
            return ChannelAttention(out_channels)
            
        elif layer_type == 'depthwise_conv':
            # Depthwise separable convolution
            return nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            
        elif layer_type == 'bottleneck':
            # Bottleneck block
            hidden_channels = out_channels // 4
            return nn.Sequential(
                nn.Conv2d(out_channels, hidden_channels, 1),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            # Standard convolution
            return nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
    
    def _create_linear_variant(self, layer_type: str, ref_layer: nn.Linear) -> nn.Module:
        """Create linear layer variant."""
        out_features = ref_layer.out_features
        
        if layer_type == 'bottleneck':
            hidden_features = out_features // 2
            return nn.Sequential(
                nn.Linear(out_features, hidden_features),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_features, out_features)
            )
        else:
            return nn.Sequential(
                nn.Linear(out_features, out_features),
                nn.ReLU(inplace=True)
            )
    
    def _insert_layer_after(
        self, 
        model: nn.Module, 
        target_name: str, 
        new_layer: nn.Module,
        new_name: str
    ):
        """Insert new layer after target layer."""
        # This is a simplified implementation
        # In practice, you'd need to handle the module tree properly
        parts = target_name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        
        # Create a sequential wrapper if needed
        old_layer = getattr(parent, parts[-1])
        if isinstance(old_layer, nn.Sequential):
            # Append to existing sequential
            old_layer.add_module(new_name, new_layer)
        else:
            # Replace with sequential containing both
            new_sequential = nn.Sequential(OrderedDict([
                ('original', old_layer),
                (new_name, new_layer)
            ]))
            setattr(parent, parts[-1], new_sequential)


class AdaptiveDataFlowOperator(StructuralOperator):
    """
    Operator that adaptively adjusts data flow sizes based on 
    feature complexity and computational efficiency.
    """
    
    def __init__(
        self,
        min_spatial_size: int = 7,
        complexity_threshold: float = 0.3,
        device: Optional[torch.device] = None
    ):
        self.min_spatial_size = min_spatial_size
        self.complexity_threshold = complexity_threshold
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def apply(self, model: nn.Module, metrics: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
        """Apply adaptive data flow adjustments."""
        modified_model = copy.deepcopy(model)
        adjustments = []
        
        # Analyze each layer's output complexity
        for name, module in modified_model.named_modules():
            if isinstance(module, nn.Conv2d):
                # Get activation complexity from metrics
                complexity = metrics.get(f'{name}_complexity', 0.5)
                
                if complexity < self.complexity_threshold:
                    # Low complexity - can reduce spatial size
                    adjustment = self._create_downsampling(module)
                    if adjustment:
                        self._insert_after_layer(modified_model, name, adjustment)
                        adjustments.append({
                            'layer': name,
                            'type': 'downsample',
                            'reason': f'low_complexity_{complexity:.3f}'
                        })
        
        return modified_model, {'adjustments': adjustments}
    
    def can_apply(self, model: nn.Module, metrics: Dict[str, Any]) -> bool:
        """Check if data flow adjustment is applicable."""
        return any(f'{name}_complexity' in metrics for name, _ in model.named_modules())
    
    def _create_downsampling(self, conv_layer: nn.Conv2d) -> Optional[nn.Module]:
        """Create appropriate downsampling layer."""
        # Use strided convolution for learnable downsampling
        return nn.Conv2d(
            conv_layer.out_channels,
            conv_layer.out_channels,
            kernel_size=3,
            stride=2,
            padding=1
        )
    
    def _insert_after_layer(self, model: nn.Module, layer_name: str, new_module: nn.Module):
        """Insert module after specified layer."""
        # Implementation would handle the module tree properly
        pass


class BranchSpecializationOperator(StructuralOperator):
    """
    Create specialized branches for different types of features.
    """
    
    def __init__(
        self,
        num_branches: int = 3,
        specializations: Optional[List[str]] = None,
        device: Optional[torch.device] = None
    ):
        self.num_branches = num_branches
        self.specializations = specializations or ['fine', 'coarse', 'global']
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def apply(self, model: nn.Module, metrics: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
        """Create specialized branches."""
        # This would implement branch creation logic
        # For now, return model unchanged
        return model, {'message': 'Branch specialization not yet implemented'}
    
    def can_apply(self, model: nn.Module, metrics: Dict[str, Any]) -> bool:
        return True


# Helper modules
class ChannelAttention(nn.Module):
    """Channel attention module for feature recalibration."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x) 