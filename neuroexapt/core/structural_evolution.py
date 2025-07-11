"""
Structural Evolution module for Neuro Exapt.

This module implements dynamic network structure optimization through
information-theoretic guided pruning and expansion.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass
import copy
import warnings
from collections import OrderedDict


@dataclass
class EvolutionStep:
    """Record of a structural evolution step."""
    epoch: int
    action: str  # 'prune', 'expand', 'mutate'
    target_layers: List[str]
    metrics_before: Dict[str, float]
    metrics_after: Dict[str, float]
    info_gain: float
    parameters_before: int
    parameters_after: int
    
    def efficiency_gain(self) -> float:
        """Calculate efficiency gain from this evolution step."""
        if self.parameters_before == 0:
            return 0.0
        param_reduction = (self.parameters_before - self.parameters_after) / self.parameters_before
        performance_change = self.metrics_after.get('accuracy', 0) - self.metrics_before.get('accuracy', 0)
        return param_reduction - performance_change  # Positive if good trade-off


class StructuralEvolution:
    """
    Structural Evolution engine for dynamic network architecture optimization.
    
    Implements information-theoretic guided structural modifications:
    - Entropy-based pruning
    - Mutual information guided expansion
    - Discrete parameter mutation
    """
    
    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        prune_ratio: float = 0.1,
        expand_ratio: float = 0.1,
        min_layers: int = 3,
        max_layers: int = 100,
        device: Optional[torch.device] = None
    ):
        """
        Initialize Structural Evolution engine.
        
        Args:
            alpha: Information retention coefficient
            beta: Structure variation coefficient
            prune_ratio: Maximum ratio of layers to prune in one step
            expand_ratio: Maximum ratio of layers to expand in one step
            min_layers: Minimum number of layers to maintain
            max_layers: Maximum number of layers allowed
            device: Torch device for computations
        """
        self.alpha = alpha
        self.beta = beta
        self.prune_ratio = prune_ratio
        self.expand_ratio = expand_ratio
        self.min_layers = min_layers
        self.max_layers = max_layers
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Evolution history
        self.evolution_history: List[EvolutionStep] = []
        self.current_epoch = 0
        
        # Layer registry
        self.layer_registry: Dict[str, nn.Module] = OrderedDict()
        self.layer_importance: Dict[str, float] = {}
        self.layer_entropy: Dict[str, float] = {}
        
    def register_model(self, model: nn.Module):
        """
        Register a model for structural evolution.
        
        Args:
            model: PyTorch model to evolve
        """
        self.layer_registry.clear()
        
        # Register all layers with learnable parameters
        for name, module in model.named_modules():
            if len(list(module.parameters())) > 0 and len(list(module.children())) == 0:
                self.layer_registry[name] = module
                
    def compute_structural_entropy(self, model: nn.Module) -> float:
        """
        Compute structural entropy of the network.
        
        Implements: S = -Σ p_i * log(p_i)
        where p_i is the normalized importance of layer i.
        
        Args:
            model: Neural network model
            
        Returns:
            Structural entropy value
        """
        if not self.layer_importance:
            return 0.0
            
        # Normalize importance values to probabilities
        total_importance = sum(self.layer_importance.values())
        if total_importance == 0:
            return 0.0
            
        probabilities = [imp / total_importance for imp in self.layer_importance.values()]
        
        # Calculate entropy
        entropy = 0.0
        for p in probabilities:
            if p > 0:
                entropy -= p * np.log(p)
                
        return entropy
    
    def calculate_entropy_gradient(
        self,
        current_entropy: float,
        target_entropy: float,
        layer_importances: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate entropy gradient for each layer.
        
        Implements: ∂S/∂t = -α * I(L_i;O) + β * KL(p_old||p_new)
        
        Args:
            current_entropy: Current structural entropy
            target_entropy: Target structural entropy
            layer_importances: Layer importance values
            
        Returns:
            Entropy gradients for each layer
        """
        gradients = {}
        
        for layer_name, importance in layer_importances.items():
            # Information retention term
            retention_term = -self.alpha * importance
            
            # Structure variation term (simplified KL divergence)
            old_p = self.layer_importance.get(layer_name, 0.5)
            new_p = importance
            kl_term = self.beta * (new_p * np.log((new_p + 1e-10) / (old_p + 1e-10)))
            
            # Combined gradient
            gradients[layer_name] = retention_term + kl_term
            
        return gradients
    
    def prune_by_entropy(
        self,
        model: nn.Module,
        layer_entropies: Dict[str, float],
        threshold: float = 0.3
    ) -> Tuple[nn.Module, List[str]]:
        """
        Prune layers based on entropy threshold.
        
        Args:
            model: Model to prune
            layer_entropies: Entropy values for each layer
            threshold: Entropy threshold for pruning
            
        Returns:
            Pruned model and list of pruned layer names
        """
        # Sort layers by entropy (ascending)
        sorted_layers = sorted(layer_entropies.items(), key=lambda x: x[1])
        
        # Determine layers to prune
        layers_to_prune = []
        current_layer_count = len(self.layer_registry)
        max_prune_count = int(current_layer_count * self.prune_ratio)
        
        for layer_name, entropy in sorted_layers:
            if entropy < threshold and len(layers_to_prune) < max_prune_count:
                # Check if we maintain minimum layers
                if current_layer_count - len(layers_to_prune) > self.min_layers:
                    layers_to_prune.append(layer_name)
                    
        if not layers_to_prune:
            return model, []
            
        # Create pruned model
        pruned_model = self._remove_layers(model, layers_to_prune)
        
        return pruned_model, layers_to_prune
    
    def expand_with_mi(
        self,
        model: nn.Module,
        layer_importances: Dict[str, float],
        gamma: float = 0.1
    ) -> Tuple[nn.Module, List[str]]:
        """
        Expand network guided by mutual information.
        
        Args:
            model: Model to expand
            layer_importances: Importance values for each layer
            gamma: Expansion threshold factor
            
        Returns:
            Expanded model and list of new layer names
        """
        # Sort layers by importance (descending)
        sorted_layers = sorted(layer_importances.items(), key=lambda x: x[1], reverse=True)
        
        # Determine expansion points
        expansion_points = []
        current_layer_count = len(self.layer_registry)
        max_expand_count = int(current_layer_count * self.expand_ratio)
        
        # Find high-importance layers as expansion candidates
        avg_importance = np.mean(list(layer_importances.values()))
        expansion_threshold = avg_importance * (1 + gamma)
        
        for layer_name, importance in sorted_layers:
            if importance > expansion_threshold and len(expansion_points) < max_expand_count:
                if current_layer_count + len(expansion_points) < self.max_layers:
                    expansion_points.append(layer_name)
                    
        if not expansion_points:
            return model, []
            
        # Create expanded model
        expanded_model, new_layers = self._add_layers(model, expansion_points)
        
        return expanded_model, new_layers
    
    def mutate_discrete_parameters(
        self,
        model: nn.Module,
        mutation_rate: float = 0.1,
        parameter_ranges: Optional[Dict[str, Tuple[int, int]]] = None
    ) -> nn.Module:
        """
        Mutate discrete parameters using continuous relaxation.
        
        Implements: k = floor(σ(θ) * (k_max - k_min) + 0.5)
        
        Args:
            model: Model to mutate
            mutation_rate: Probability of mutating each parameter
            parameter_ranges: Optional ranges for discrete parameters
            
        Returns:
            Mutated model
        """
        if parameter_ranges is None:
            parameter_ranges = {
                'kernel_size': (1, 7),
                'stride': (1, 3),
                'dilation': (1, 3),
                'groups': (1, 8)
            }
            
        mutated_model = copy.deepcopy(model)
        
        for name, module in mutated_model.named_modules():
            if np.random.random() < mutation_rate:
                # Mutate convolutional layers
                if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                    self._mutate_conv_layer(module, parameter_ranges)
                # Mutate other layer types as needed
                elif isinstance(module, nn.Linear):
                    self._mutate_linear_layer(module)
                    
        return mutated_model
    
    def evolve_step(
        self,
        model: nn.Module,
        info_theory_metrics: Dict[str, Any],
        entropy_metrics: Dict[str, Any],
        performance_metrics: Dict[str, float]
    ) -> Tuple[nn.Module, EvolutionStep]:
        """
        Perform one evolution step based on current metrics.
        
        Args:
            model: Current model
            info_theory_metrics: Information theory metrics
            entropy_metrics: Entropy-related metrics
            performance_metrics: Performance metrics (accuracy, loss, etc.)
            
        Returns:
            Evolved model and evolution step record
        """
        self.current_epoch += 1
        
        # Extract relevant metrics
        layer_importances = info_theory_metrics.get('layer_importances', {})
        layer_entropies = entropy_metrics.get('layer_entropies', {})
        current_entropy = entropy_metrics.get('current_entropy', 0.0)
        threshold = entropy_metrics.get('threshold', 0.3)
        
        # Update internal state
        self.layer_importance = layer_importances
        self.layer_entropy = layer_entropies
        
        # Count parameters before evolution
        params_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Decide evolution action
        action = self._decide_action(current_entropy, threshold, performance_metrics)
        
        # Execute evolution
        if action == 'prune':
            evolved_model, affected_layers = self.prune_by_entropy(
                model, layer_entropies, threshold
            )
        elif action == 'expand':
            evolved_model, affected_layers = self.expand_with_mi(
                model, layer_importances
            )
        elif action == 'mutate':
            evolved_model = self.mutate_discrete_parameters(model)
            affected_layers = ['discrete_parameters']
        else:
            evolved_model = model
            affected_layers = []
            
        # Count parameters after evolution
        params_after = sum(p.numel() for p in evolved_model.parameters() if p.requires_grad)
        
        # Calculate information gain
        info_gain = self._calculate_info_gain(model, evolved_model, layer_importances)
        
        # Create evolution record
        step = EvolutionStep(
            epoch=self.current_epoch,
            action=action,
            target_layers=affected_layers,
            metrics_before=performance_metrics.copy(),
            metrics_after=performance_metrics.copy(),  # Will be updated after training
            info_gain=info_gain,
            parameters_before=params_before,
            parameters_after=params_after
        )
        
        self.evolution_history.append(step)
        
        return evolved_model, step
    
    def _decide_action(
        self,
        current_entropy: float,
        threshold: float,
        performance_metrics: Dict[str, float]
    ) -> str:
        """Decide which evolution action to take."""
        val_accuracy = performance_metrics.get('val_accuracy', 0)
        
        # More aggressive expansion for better accuracy
        if val_accuracy < 85.0:  # If accuracy is below 85%, prioritize expansion
            if self.current_epoch % 5 == 0:  # More frequent expansion
                return 'expand'
            elif current_entropy > threshold * 1.2:  # Lower threshold for expansion
                return 'expand'
            elif self.current_epoch % 8 == 0:  # Periodic mutation
                return 'mutate'
            else:
                return 'none'
        else:
            # Normal evolution logic for high accuracy
            # Check performance degradation
            if len(self.evolution_history) > 0:
                last_step = self.evolution_history[-1]
                perf_drop = performance_metrics.get('accuracy', 0) - last_step.metrics_before.get('accuracy', 0)
                
                if perf_drop < -0.05:  # Significant performance drop
                    return 'expand'  # Recover by expanding
                    
            # Entropy-based decision
            if current_entropy < threshold * 0.8:
                return 'prune'
            elif current_entropy > threshold * 1.5:
                return 'expand'
            elif self.current_epoch % 10 == 0:  # Periodic mutation
                return 'mutate'
            else:
                return 'none'
            
    def _remove_layers(self, model: nn.Module, layers_to_remove: List[str]) -> nn.Module:
        """Remove specified layers from model."""
        # This is a simplified implementation
        # In practice, you'd need to handle skip connections, etc.
        
        pruned_model = copy.deepcopy(model)
        
        for layer_name in layers_to_remove:
            # Replace layer with identity
            parts = layer_name.split('.')
            parent = pruned_model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], nn.Identity())
            
        return pruned_model
        
    def _add_layers(
        self,
        model: nn.Module,
        expansion_points: List[str]
    ) -> Tuple[nn.Module, List[str]]:
        """Add new layers at specified expansion points."""
        expanded_model = copy.deepcopy(model)
        new_layers = []
        
        # Get device from model
        device = next(expanded_model.parameters()).device
        
        # Check if model has special expansion methods
        if hasattr(expanded_model, 'add_expansion_layer'):
            # Use model's built-in expansion method
            for i, point in enumerate(expansion_points):
                if hasattr(expanded_model, 'get_next_layer_name'):
                    layer_name = expanded_model.get_next_layer_name()
                else:
                    layer_name = f'expanded_{i}'
                if expanded_model.add_expansion_layer(layer_name, device=device):
                    new_layers.append(layer_name)
        else:
            # Generic expansion for models without special methods
            for point in expansion_points:
                parts = point.split('.')
                parent = expanded_model
                
                # Navigate to the parent module
                for part in parts[:-1]:
                    if hasattr(parent, part):
                        parent = getattr(parent, part)
                    else:
                        continue
                
                # Get the original layer
                if not hasattr(parent, parts[-1]):
                    continue
                    
                original_layer = getattr(parent, parts[-1])
                
                # Create appropriate expansion based on layer type
                if isinstance(original_layer, nn.Conv2d):
                    # For Conv2d, add a parallel conv layer
                    in_channels = original_layer.in_channels
                    out_channels = original_layer.out_channels
                    kernel_size = original_layer.kernel_size
                    
                    # Handle kernel_size which can be int or tuple
                    if isinstance(kernel_size, int):
                        padding = kernel_size // 2
                    else:
                        padding = kernel_size[0] // 2
                    
                    # Create a sequential block with the original layer and new layer
                    new_block = nn.Sequential(
                        original_layer,
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True)
                    ).to(device)
                    
                    setattr(parent, parts[-1], new_block)
                    new_layers.append(f"{point}_expansion")
                    
                elif isinstance(original_layer, nn.Linear):
                    # For Linear layers, add intermediate layer
                    in_features = original_layer.in_features
                    out_features = original_layer.out_features
                    hidden_features = int((in_features + out_features) * 0.75)
                    
                    new_block = nn.Sequential(
                        nn.Linear(in_features, hidden_features),
                        nn.BatchNorm1d(hidden_features),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.1),
                        nn.Linear(hidden_features, out_features)
                    ).to(device)
                    
                    # Copy weights from original to maintain some behavior
                    with torch.no_grad():
                        # Initialize first layer to approximate original transformation
                        min_features = min(out_features, hidden_features)
                        new_block[0].weight.data[:min_features, :] = original_layer.weight.data[:min_features, :]
                        if original_layer.bias is not None and new_block[0].bias is not None:
                            new_block[0].bias.data[:min_features] = original_layer.bias.data[:min_features]
                    
                    setattr(parent, parts[-1], new_block)
                    new_layers.append(f"{point}_expansion")
        
        return expanded_model, new_layers
        
    def _mutate_conv_layer(self, layer: nn.Module, ranges: Dict[str, Tuple[int, int]]):
        """Mutate convolutional layer parameters."""
        # This would involve changing kernel size, stride, etc.
        # Simplified for this implementation
        pass
        
    def _mutate_linear_layer(self, layer: nn.Module):
        """Mutate linear layer parameters."""
        # This could involve changing hidden dimensions
        # Simplified for this implementation
        pass
        
    def _calculate_info_gain(
        self,
        old_model: nn.Module,
        new_model: nn.Module,
        layer_importances: Dict[str, float]
    ) -> float:
        """Calculate information gain from evolution."""
        # Simplified calculation based on preserved importance
        old_importance = sum(layer_importances.values())
        
        # Estimate new importance (simplified)
        new_importance = old_importance * 0.95  # Assume slight loss
        
        return new_importance - old_importance
        
    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get summary of evolution history."""
        if not self.evolution_history:
            return {}
            
        total_steps = len(self.evolution_history)
        prune_steps = sum(1 for s in self.evolution_history if s.action == 'prune')
        expand_steps = sum(1 for s in self.evolution_history if s.action == 'expand')
        mutate_steps = sum(1 for s in self.evolution_history if s.action == 'mutate')
        
        total_param_reduction = 0
        total_info_gain = 0
        
        for step in self.evolution_history:
            total_param_reduction += step.parameters_before - step.parameters_after
            total_info_gain += step.info_gain
            
        return {
            'total_evolution_steps': total_steps,
            'prune_steps': prune_steps,
            'expand_steps': expand_steps,
            'mutate_steps': mutate_steps,
            'total_parameter_reduction': total_param_reduction,
            'total_information_gain': total_info_gain,
            'average_efficiency_gain': np.mean([s.efficiency_gain() for s in self.evolution_history])
        }
        
    def save_evolution_history(self, filepath: str):
        """Save evolution history to file."""
        import json
        
        history_data = []
        for step in self.evolution_history:
            history_data.append({
                'epoch': step.epoch,
                'action': step.action,
                'target_layers': step.target_layers,
                'metrics_before': step.metrics_before,
                'metrics_after': step.metrics_after,
                'info_gain': step.info_gain,
                'parameters_before': step.parameters_before,
                'parameters_after': step.parameters_after,
                'efficiency_gain': step.efficiency_gain()
            })
            
        with open(filepath, 'w') as f:
            json.dump(history_data, f, indent=2) 