"""
Structural Operators module for Neuro Exapt.

This module implements various operators for network structure modification:
- PruneByEntropy: Entropy-based layer pruning
- ExpandWithMI: Mutual information guided expansion
- MutateDiscrete: Discrete parameter mutation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from abc import ABC, abstractmethod
import copy
from collections import OrderedDict


class StructuralOperator(ABC):
    """Base class for all structural operators."""
    
    @abstractmethod
    def apply(self, model: nn.Module, metrics: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Apply the operator to a model.
        
        Args:
            model: Input model
            metrics: Metrics to guide the operation
            
        Returns:
            Modified model and operation info
        """
        pass
    
    @abstractmethod
    def can_apply(self, model: nn.Module, metrics: Dict[str, Any]) -> bool:
        """Check if the operator can be applied."""
        pass


class PruneByEntropy(StructuralOperator):
    """
    Prune layers based on entropy threshold.
    
    Low entropy indicates redundant or uninformative layers.
    """
    
    def __init__(
        self,
        threshold: float = 0.3,
        min_layers_to_keep: int = 3,
        prune_ratio: float = 0.1,
        layer_types: Optional[List[type]] = None
    ):
        """
        Initialize entropy-based pruning operator.
        
        Args:
            threshold: Entropy threshold below which layers are pruned
            min_layers_to_keep: Minimum number of layers to retain
            prune_ratio: Maximum ratio of layers to prune in one step
            layer_types: Types of layers eligible for pruning
        """
        self.threshold = threshold
        self.min_layers_to_keep = min_layers_to_keep
        self.prune_ratio = prune_ratio
        self.layer_types = layer_types or [nn.Conv2d, nn.Linear, nn.Conv1d, nn.Conv3d]
        
    def apply(self, model: nn.Module, metrics: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
        """Apply entropy-based pruning."""
        layer_entropies = metrics.get('layer_entropies', {})
        if not layer_entropies:
            return model, {'pruned_layers': [], 'message': 'No entropy metrics available'}
            
        # Get eligible layers
        eligible_layers = self._get_eligible_layers(model)
        
        # Sort by entropy (ascending)
        entropy_items = [(name, layer_entropies.get(name, float('inf'))) 
                        for name in eligible_layers]
        entropy_items.sort(key=lambda x: x[1])
        
        # Determine layers to prune
        max_prune = int(len(eligible_layers) * self.prune_ratio)
        layers_to_prune = []
        
        for layer_name, entropy in entropy_items:
            if entropy < self.threshold and len(layers_to_prune) < max_prune:
                if len(eligible_layers) - len(layers_to_prune) > self.min_layers_to_keep:
                    layers_to_prune.append(layer_name)
                    
        if not layers_to_prune:
            return model, {'pruned_layers': [], 'message': 'No layers meet pruning criteria'}
            
        # Create pruned model
        pruned_model = self._prune_layers(model, layers_to_prune)
        
        info = {
            'pruned_layers': layers_to_prune,
            'num_pruned': len(layers_to_prune),
            'threshold_used': self.threshold,
            'lowest_entropy': entropy_items[0][1] if entropy_items else None
        }
        
        return pruned_model, info
        
    def can_apply(self, model: nn.Module, metrics: Dict[str, Any]) -> bool:
        """Check if pruning can be applied."""
        layer_entropies = metrics.get('layer_entropies', {})
        if not layer_entropies:
            return False
            
        eligible_layers = self._get_eligible_layers(model)
        if len(eligible_layers) <= self.min_layers_to_keep:
            return False
            
        # Check if any layer has entropy below threshold
        for name in eligible_layers:
            if name in layer_entropies and layer_entropies[name] < self.threshold:
                return True
                
        return False
        
    def _get_eligible_layers(self, model: nn.Module) -> List[str]:
        """Get list of layers eligible for pruning."""
        eligible = []
        
        for name, module in model.named_modules():
            if any(isinstance(module, layer_type) for layer_type in self.layer_types):
                if len(list(module.parameters())) > 0:
                    eligible.append(name)
                    
        return eligible
        
    def _prune_layers(self, model: nn.Module, layers_to_prune: List[str]) -> nn.Module:
        """Replace specified layers with identity mapping."""
        pruned_model = copy.deepcopy(model)
        
        for layer_name in layers_to_prune:
            parts = layer_name.split('.')
            parent = pruned_model
            
            # Navigate to parent module
            for part in parts[:-1]:
                parent = getattr(parent, part)
                
            # Replace with identity
            setattr(parent, parts[-1], nn.Identity())
            
        return pruned_model


class ExpandWithMI(StructuralOperator):
    """
    Expand network capacity guided by mutual information.
    
    High mutual information indicates information bottlenecks.
    """
    
    def __init__(
        self,
        gamma: float = 0.1,
        max_layers_to_add: int = 5,
        expand_ratio: float = 0.1,
        expansion_factor: float = 1.5
    ):
        """
        Initialize MI-guided expansion operator.
        
        Args:
            gamma: Threshold factor for determining expansion points
            max_layers_to_add: Maximum layers to add in one step
            expand_ratio: Maximum ratio of layers to expand
            expansion_factor: Factor for new layer dimensions
        """
        self.gamma = gamma
        self.max_layers_to_add = max_layers_to_add
        self.expand_ratio = expand_ratio
        self.expansion_factor = expansion_factor
        
    def apply(self, model: nn.Module, metrics: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
        """Apply MI-guided expansion."""
        layer_importances = metrics.get('layer_importances', {})
        if not layer_importances:
            return model, {'expanded_layers': [], 'message': 'No importance metrics available'}
            
        # Calculate expansion threshold
        importances = list(layer_importances.values())
        if not importances:
            return model, {'expanded_layers': [], 'message': 'No layer importances found'}
            
        avg_importance = np.mean(importances)
        threshold = avg_importance * (1 + self.gamma)
        
        # Find expansion candidates
        candidates = []
        for name, importance in layer_importances.items():
            if importance > threshold:
                candidates.append((name, importance))
                
        # Sort by importance (descending)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Select layers to expand
        max_expand = min(
            self.max_layers_to_add,
            int(len(layer_importances) * self.expand_ratio)
        )
        
        layers_to_expand = [name for name, _ in candidates[:max_expand]]
        
        if not layers_to_expand:
            return model, {'expanded_layers': [], 'message': 'No layers meet expansion criteria'}
            
        # Create expanded model
        expanded_model = self._expand_layers(model, layers_to_expand)
        
        info = {
            'expanded_layers': layers_to_expand,
            'num_expanded': len(layers_to_expand),
            'threshold_used': threshold,
            'highest_importance': candidates[0][1] if candidates else None
        }
        
        return expanded_model, info
        
    def can_apply(self, model: nn.Module, metrics: Dict[str, Any]) -> bool:
        """Check if expansion can be applied."""
        layer_importances = metrics.get('layer_importances', {})
        if not layer_importances:
            return False
            
        importances = list(layer_importances.values())
        if not importances:
            return False
            
        avg_importance = np.mean(importances)
        threshold = avg_importance * (1 + self.gamma)
        
        # Check if any layer exceeds threshold
        return any(imp > threshold for imp in importances)
        
    def _expand_layers(self, model: nn.Module, layers_to_expand: List[str]) -> nn.Module:
        """Add new layers at specified expansion points."""
        expanded_model = copy.deepcopy(model)
        
        for layer_name in layers_to_expand:
            parts = layer_name.split('.')
            parent = expanded_model
            
            # Navigate to target module
            for part in parts[:-1]:
                parent = getattr(parent, part)
                
            original_layer = getattr(parent, parts[-1])
            
            # Create expanded block based on layer type
            if isinstance(original_layer, nn.Linear):
                expanded_block = self._create_linear_expansion(original_layer)
            elif isinstance(original_layer, nn.Conv2d):
                expanded_block = self._create_conv2d_expansion(original_layer)
            else:
                # Skip unsupported layer types
                continue
                
            # Replace with expanded block
            setattr(parent, parts[-1], expanded_block)
            
        return expanded_model
        
    def _create_linear_expansion(self, layer: nn.Linear) -> nn.Module:
        """Create expanded linear block."""
        in_features = layer.in_features
        out_features = layer.out_features
        hidden_features = int(out_features * self.expansion_factor)
        
        return nn.Sequential(
            layer,
            nn.ReLU(inplace=True),
            nn.Linear(out_features, hidden_features),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_features, out_features)
        )
        
    def _create_conv2d_expansion(self, layer: nn.Conv2d) -> nn.Module:
        """Create expanded conv2d block."""
        in_channels = layer.in_channels
        out_channels = layer.out_channels
        hidden_channels = int(out_channels * self.expansion_factor)
        
        return nn.Sequential(
            layer,
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, hidden_channels,
                kernel_size=3, padding=1, stride=1
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_channels, out_channels,
                kernel_size=1, stride=1
            )
        )


class MutateDiscrete(StructuralOperator):
    """
    Mutate discrete parameters using continuous relaxation.
    
    Implements differentiable discrete parameter optimization.
    """
    
    def __init__(
        self,
        mutation_rate: float = 0.1,
        temperature: float = 1.0,
        parameter_ranges: Optional[Dict[str, Tuple[int, int]]] = None
    ):
        """
        Initialize discrete parameter mutation operator.
        
        Args:
            mutation_rate: Probability of mutating each parameter
            temperature: Temperature for Gumbel-softmax relaxation
            parameter_ranges: Ranges for discrete parameters
        """
        self.mutation_rate = mutation_rate
        self.temperature = temperature
        self.parameter_ranges = parameter_ranges or {
            'kernel_size': (1, 7),
            'stride': (1, 3),
            'dilation': (1, 3),
            'groups': (1, 16),
            'num_heads': (1, 16),  # For attention layers
            'hidden_ratio': (1, 4)  # For MLP expansion
        }
        
    def apply(self, model: nn.Module, metrics: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
        """Apply discrete parameter mutation."""
        mutated_model = copy.deepcopy(model)
        mutations = []
        
        for name, module in mutated_model.named_modules():
            if np.random.random() < self.mutation_rate:
                mutation_info = self._mutate_module(name, module)
                if mutation_info:
                    mutations.append(mutation_info)
                    
        info = {
            'mutations': mutations,
            'num_mutations': len(mutations),
            'mutation_rate_used': self.mutation_rate
        }
        
        return mutated_model, info
        
    def can_apply(self, model: nn.Module, metrics: Dict[str, Any]) -> bool:
        """Mutation can always be applied."""
        return True
        
    def _mutate_module(self, name: str, module: nn.Module) -> Optional[Dict[str, Any]]:
        """Mutate a single module's discrete parameters."""
        mutation_info = {'layer': name, 'type': type(module).__name__, 'changes': {}}
        
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            # Mutate convolution parameters
            changes = self._mutate_conv_params(module)
            if changes:
                mutation_info['changes'] = changes
                return mutation_info
                
        elif isinstance(module, nn.MultiheadAttention):
            # Mutate attention parameters
            changes = self._mutate_attention_params(module)
            if changes:
                mutation_info['changes'] = changes
                return mutation_info
                
        return None
        
    def _mutate_conv_params(self, conv: nn.Module) -> Dict[str, Tuple[Any, Any]]:
        """Mutate convolutional layer parameters."""
        changes = {}
        
        # Note: Actual mutation of kernel size, stride, etc. requires
        # reconstructing the layer, which is complex. This is a simplified
        # version that tracks what would be changed.
        
        if 'kernel_size' in self.parameter_ranges and np.random.random() < 0.5:
            old_kernel = conv.kernel_size
            k_min, k_max = self.parameter_ranges['kernel_size']
            
            # Continuous relaxation
            theta = np.random.randn()
            k_continuous = torch.sigmoid(torch.tensor(theta)) * (k_max - k_min) + k_min
            new_kernel = int(k_continuous.item() + 0.5)
            
            if isinstance(old_kernel, tuple):
                new_kernel = (new_kernel, new_kernel)
                
            changes['kernel_size'] = (old_kernel, new_kernel)
            
        return changes
        
    def _mutate_attention_params(self, attn: nn.MultiheadAttention) -> Dict[str, Tuple[Any, Any]]:
        """Mutate attention layer parameters."""
        changes = {}
        
        if 'num_heads' in self.parameter_ranges and np.random.random() < 0.5:
            old_heads = attn.num_heads
            h_min, h_max = self.parameter_ranges['num_heads']
            
            # Ensure new number of heads divides embed_dim
            embed_dim = attn.embed_dim
            valid_heads = [h for h in range(h_min, h_max + 1) if embed_dim % h == 0]
            
            if valid_heads:
                theta = np.random.randn()
                h_continuous = torch.sigmoid(torch.tensor(theta)) * len(valid_heads)
                new_heads_idx = int(h_continuous.item())
                new_heads = valid_heads[min(new_heads_idx, len(valid_heads) - 1)]
                
                changes['num_heads'] = (old_heads, new_heads)
                
        return changes


class CompoundOperator(StructuralOperator):
    """
    Combine multiple operators with conditional logic.
    """
    
    def __init__(self, operators: List[StructuralOperator]):
        """
        Initialize compound operator.
        
        Args:
            operators: List of operators to combine
        """
        self.operators = operators
        
    def apply(self, model: nn.Module, metrics: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
        """Apply operators sequentially based on conditions."""
        current_model = model
        all_info = {'operations': []}
        
        for i, operator in enumerate(self.operators):
            if operator.can_apply(current_model, metrics):
                current_model, info = operator.apply(current_model, metrics)
                all_info['operations'].append({
                    'operator': type(operator).__name__,
                    'index': i,
                    'info': info
                })
                
        all_info['num_operations'] = len(all_info['operations'])
        
        return current_model, all_info
        
    def can_apply(self, model: nn.Module, metrics: Dict[str, Any]) -> bool:
        """Check if any operator can be applied."""
        return any(op.can_apply(model, metrics) for op in self.operators)


class AdaptiveOperator(StructuralOperator):
    """
    Operator that adapts its parameters based on training progress.
    """
    
    def __init__(
        self,
        base_operator: StructuralOperator,
        schedule: str = 'linear',
        initial_scale: float = 1.0,
        final_scale: float = 0.1
    ):
        """
        Initialize adaptive operator.
        
        Args:
            base_operator: Base operator to adapt
            schedule: Adaptation schedule ('linear', 'cosine', 'exponential')
            initial_scale: Initial parameter scale
            final_scale: Final parameter scale
        """
        self.base_operator = base_operator
        self.schedule = schedule
        self.initial_scale = initial_scale
        self.final_scale = final_scale
        self.current_epoch = 0
        self.total_epochs = 100  # Default, should be set externally
        
    def update_epoch(self, epoch: int, total_epochs: int):
        """Update current epoch for adaptation."""
        self.current_epoch = epoch
        self.total_epochs = total_epochs
        self._adapt_parameters()
        
    def _adapt_parameters(self):
        """Adapt operator parameters based on schedule."""
        progress = self.current_epoch / self.total_epochs
        
        if self.schedule == 'linear':
            scale = self.initial_scale + (self.final_scale - self.initial_scale) * progress
        elif self.schedule == 'cosine':
            scale = self.final_scale + 0.5 * (self.initial_scale - self.final_scale) * \
                   (1 + np.cos(np.pi * progress))
        elif self.schedule == 'exponential':
            scale = self.initial_scale * (self.final_scale / self.initial_scale) ** progress
        else:
            scale = self.initial_scale
            
        # Apply scale to base operator parameters
        if isinstance(self.base_operator, PruneByEntropy):
            self.base_operator.threshold *= scale
        elif isinstance(self.base_operator, ExpandWithMI):
            self.base_operator.gamma *= scale
        elif isinstance(self.base_operator, MutateDiscrete):
            self.base_operator.mutation_rate *= scale
            
    def apply(self, model: nn.Module, metrics: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
        """Apply base operator with adapted parameters."""
        return self.base_operator.apply(model, metrics)
        
    def can_apply(self, model: nn.Module, metrics: Dict[str, Any]) -> bool:
        """Check if base operator can be applied."""
        return self.base_operator.can_apply(model, metrics) 