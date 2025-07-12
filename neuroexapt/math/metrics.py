"""
Mathematical metrics module for Neuro Exapt.

This module implements various information-theoretic and complexity metrics
for neural network analysis and optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from scipy import stats
from scipy.special import softmax
import warnings


def calculate_entropy(
    tensor: torch.Tensor,
    method: str = 'histogram',
    bins: int = 30,
    epsilon: float = 1e-10
) -> float:
    """
    Calculate Shannon entropy of a tensor.
    
    Args:
        tensor: Input tensor
        method: Method for entropy calculation ('histogram', 'kde', 'discrete')
        bins: Number of bins for histogram method
        epsilon: Small value to avoid log(0)
        
    Returns:
        Entropy value in bits
    """
    data = tensor.detach().cpu().numpy().flatten()
    
    if method == 'histogram':
        # Histogram-based entropy estimation
        hist, _ = np.histogram(data, bins=bins)
        hist = hist + epsilon
        hist = hist / hist.sum()
        return -np.sum(hist * np.log2(hist))
        
    elif method == 'kde':
        # Kernel density estimation
        try:
            kde = stats.gaussian_kde(data)
            # Sample from KDE for entropy estimation
            samples = kde.resample(1000).flatten()
            hist, _ = np.histogram(samples, bins=bins)
            hist = hist + epsilon
            hist = hist / hist.sum()
            return -np.sum(hist * np.log2(hist))
        except:
            # Fallback to histogram if KDE fails
            return calculate_entropy(tensor, method='histogram', bins=bins)
            
    elif method == 'discrete':
        # For discrete distributions (e.g., after softmax)
        if tensor.dim() > 1:
            # Apply softmax if needed
            probs = F.softmax(tensor.view(-1), dim=0)
        else:
            probs = tensor.view(-1)
        probs = probs.clamp(min=epsilon)
        return -torch.sum(probs * torch.log2(probs)).item()
        
    else:
        raise ValueError(f"Unknown entropy calculation method: {method}")


def calculate_mutual_information(
    x: torch.Tensor,
    y: torch.Tensor,
    method: str = 'histogram',
    bins: int = 30
) -> float:
    """
    Calculate mutual information I(X;Y).
    
    Args:
        x: First variable
        y: Second variable
        method: Method for MI calculation
        bins: Number of bins for discretization
        
    Returns:
        Mutual information in bits
    """
    x_data = x.detach().cpu().numpy().flatten()
    y_data = y.detach().cpu().numpy().flatten()
    
    if len(x_data) != len(y_data):
        raise ValueError("X and Y must have the same number of samples")
    
    if method == 'histogram':
        # 2D histogram for joint probability
        hist_2d, _, _ = np.histogram2d(x_data, y_data, bins=bins)
        hist_2d = hist_2d + 1e-10
        
        # Joint probability
        p_xy = hist_2d / hist_2d.sum()
        
        # Marginal probabilities
        p_x = p_xy.sum(axis=1)
        p_y = p_xy.sum(axis=0)
        
        # MI = H(X) + H(Y) - H(X,Y)
        h_x = -np.sum(p_x * np.log2(p_x + 1e-10))
        h_y = -np.sum(p_y * np.log2(p_y + 1e-10))
        h_xy = -np.sum(p_xy * np.log2(p_xy + 1e-10))
        
        return h_x + h_y - h_xy
        
    else:
        raise ValueError(f"Unknown MI calculation method: {method}")


def calculate_conditional_entropy(
    x: torch.Tensor,
    y: torch.Tensor,
    bins: int = 30
) -> float:
    """
    Calculate conditional entropy H(X|Y).
    
    Args:
        x: Variable X
        y: Conditioning variable Y
        bins: Number of bins for discretization
        
    Returns:
        Conditional entropy in bits
    """
    # H(X|Y) = H(X,Y) - H(Y)
    x_data = x.detach().cpu().numpy().flatten()
    y_data = y.detach().cpu().numpy().flatten()
    
    # Joint entropy
    hist_2d, _, _ = np.histogram2d(x_data, y_data, bins=bins)
    hist_2d = hist_2d + 1e-10
    p_xy = hist_2d / hist_2d.sum()
    h_xy = -np.sum(p_xy * np.log2(p_xy + 1e-10))
    
    # Marginal entropy of Y
    p_y = p_xy.sum(axis=0)
    h_y = -np.sum(p_y * np.log2(p_y + 1e-10))
    
    return h_xy - h_y


def calculate_kl_divergence(
    p: torch.Tensor,
    q: torch.Tensor,
    epsilon: float = 1e-10
) -> float:
    """
    Calculate Kullback-Leibler divergence KL(P||Q).
    
    Args:
        p: Distribution P
        q: Distribution Q
        epsilon: Small value to avoid log(0)
        
    Returns:
        KL divergence in bits
    """
    # Ensure distributions
    if p.dim() > 1:
        p = F.softmax(p.view(-1), dim=0)
        q = F.softmax(q.view(-1), dim=0)
    
    p = p.clamp(min=epsilon)
    q = q.clamp(min=epsilon)
    
    return torch.sum(p * torch.log2(p / q)).item()


def calculate_js_divergence(
    p: torch.Tensor,
    q: torch.Tensor
) -> float:
    """
    Calculate Jensen-Shannon divergence.
    
    Args:
        p: Distribution P
        q: Distribution Q
        
    Returns:
        JS divergence in bits
    """
    # JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M), where M = 0.5 * (P + Q)
    m = 0.5 * (p + q)
    
    js = 0.5 * calculate_kl_divergence(p, m) + 0.5 * calculate_kl_divergence(q, m)
    
    return js


def calculate_layer_complexity(
    layer: nn.Module,
    include_bias: bool = True
) -> Dict[str, float]:
    """
    Calculate complexity metrics for a layer.
    
    Args:
        layer: Neural network layer
        include_bias: Whether to include bias in parameter count
        
    Returns:
        Dictionary of complexity metrics
    """
    metrics = {}
    
    # Parameter count
    total_params = 0
    trainable_params = 0
    
    for name, param in layer.named_parameters():
        if not include_bias and 'bias' in name:
            continue
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    metrics['total_parameters'] = total_params
    metrics['trainable_parameters'] = trainable_params
    
    # Layer-specific metrics
    if isinstance(layer, nn.Conv2d):
        metrics['kernel_size'] = layer.kernel_size
        metrics['in_channels'] = layer.in_channels
        metrics['out_channels'] = layer.out_channels
        metrics['groups'] = layer.groups
        metrics['computational_cost'] = (
            layer.kernel_size[0] * layer.kernel_size[1] * 
            layer.in_channels * layer.out_channels / layer.groups
        )
    elif isinstance(layer, nn.Linear):
        metrics['in_features'] = layer.in_features
        metrics['out_features'] = layer.out_features
        metrics['computational_cost'] = layer.in_features * layer.out_features
    
    return metrics


def calculate_network_complexity(
    model: nn.Module,
    input_shape: Optional[Tuple[int, ...]] = None
) -> Dict[str, Any]:
    """
    Calculate comprehensive complexity metrics for entire network.
    
    Args:
        model: Neural network model
        input_shape: Input shape for FLOPs calculation
        
    Returns:
        Dictionary of network complexity metrics
    """
    metrics = {
        'total_parameters': 0,
        'trainable_parameters': 0,
        'layer_complexities': {},
        'depth': 0,
        'width_stats': {}
    }
    
    # Count parameters and analyze layers
    layer_widths = []
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            layer_metrics = calculate_layer_complexity(module)
            if layer_metrics['total_parameters'] > 0:
                metrics['layer_complexities'][name] = layer_metrics
                metrics['total_parameters'] += layer_metrics['total_parameters']
                metrics['trainable_parameters'] += layer_metrics['trainable_parameters']
                metrics['depth'] += 1
                
                # Track width
                if 'out_features' in layer_metrics:
                    layer_widths.append(layer_metrics['out_features'])
                elif 'out_channels' in layer_metrics:
                    layer_widths.append(layer_metrics['out_channels'])
    
    # Width statistics
    if layer_widths:
        metrics['width_stats'] = {
            'mean': np.mean(layer_widths),
            'std': np.std(layer_widths),
            'min': min(layer_widths),
            'max': max(layer_widths)
        }
    
    # Estimate FLOPs if input shape provided
    if input_shape is not None:
        try:
            metrics['estimated_flops'] = estimate_flops(model, input_shape)
        except:
            warnings.warn("Could not estimate FLOPs")
    
    return metrics


def estimate_flops(
    model: nn.Module,
    input_shape: Tuple[int, ...]
) -> int:
    """
    Estimate FLOPs (floating point operations) for a model.
    
    This is a simplified estimation focusing on major operations.
    
    Args:
        model: Neural network model
        input_shape: Input shape (excluding batch dimension)
        
    Returns:
        Estimated FLOPs
    """
    total_flops = 0
    
    # Create dummy input
    dummy_input = torch.randn(1, *input_shape)
    
    # Get the actual model if it's wrapped
    actual_model = model
    if hasattr(model, 'model') and isinstance(model.model, nn.Module):
        actual_model = model.model
    elif hasattr(model, 'module') and isinstance(model.module, nn.Module):
        actual_model = model.module
    
    # Move dummy input to same device as model
    try:
        device = next(actual_model.parameters()).device
        dummy_input = dummy_input.to(device)
    except (StopIteration, AttributeError):
        # No parameters in model or not a valid module
        device = torch.device('cpu')
        dummy_input = dummy_input.to(device)
    
    # Hook to capture layer inputs/outputs
    def hook_fn(module, input, output):
        nonlocal total_flops
        
        if isinstance(module, nn.Conv2d):
            # Conv2d FLOPs: 
            # output_size * (kernel_ops * in_channels / groups + bias)
            batch_size = output.shape[0]
            out_h, out_w = output.shape[2:]
            kernel_ops = module.kernel_size[0] * module.kernel_size[1]
            bias_ops = 1 if module.bias is not None else 0
            
            flops = batch_size * (kernel_ops * module.in_channels / module.groups + bias_ops) * \
                    module.out_channels * out_h * out_w
            total_flops += flops
            
        elif isinstance(module, nn.Linear):
            # Linear FLOPs: batch_size * (in_features * out_features + bias)
            batch_size = output.shape[0]
            flops = batch_size * module.in_features * module.out_features
            if module.bias is not None:
                flops += batch_size * module.out_features
            total_flops += flops
            
        elif isinstance(module, nn.BatchNorm2d):
            # BatchNorm FLOPs: 2 * num_features * spatial_size (mean + var)
            batch_size = output.shape[0]
            num_features = output.shape[1]
            spatial_size = output.shape[2] * output.shape[3]
            flops = batch_size * 2 * num_features * spatial_size
            total_flops += flops
    
    # Register hooks
    hooks = []
    for module in actual_model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
            hooks.append(module.register_forward_hook(hook_fn))
    
    # Forward pass
    try:
        with torch.no_grad():
            actual_model.eval()
            if hasattr(model, 'model'):  # If wrapped, use wrapper's forward
                _ = model(dummy_input)
            else:
                _ = actual_model(dummy_input)
    except Exception as e:
        warnings.warn(f"Could not estimate FLOPs due to forward pass error: {str(e)}")
        total_flops = 0
    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()
    
    return int(total_flops)


def calculate_structural_entropy(
    importances: Dict[str, float]
) -> float:
    """
    Calculate structural entropy based on layer importances.
    
    Args:
        importances: Dictionary of layer importances
        
    Returns:
        Structural entropy in bits
    """
    if not importances:
        return 0.0
    
    # Normalize to probabilities
    values = list(importances.values())
    total = sum(values)
    
    if total == 0:
        return 0.0
    
    probs = [v / total for v in values]
    
    # Calculate entropy
    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * np.log2(p)
    
    return entropy


def calculate_redundancy_score(
    layer_importances: List[float],
    output_entropy: float,
    depth: int,
    depth_decay: float = 0.03
) -> float:
    """
    Calculate network redundancy score.
    
    Implements: R = 1 - Σ I(L_i;O) / (H(O) · exp(-λ · Depth))
    
    Args:
        layer_importances: List of layer importance values
        output_entropy: Output entropy H(O)
        depth: Network depth
        depth_decay: Decay factor λ
        
    Returns:
        Redundancy score between 0 and 1
    """
    if not layer_importances or output_entropy == 0:
        return 0.0
    
    total_importance = sum(layer_importances)
    normalization = output_entropy * np.exp(-depth_decay * depth)
    
    redundancy = 1 - (total_importance / (normalization + 1e-10))
    
    return max(0, min(1, redundancy))


def calculate_discrete_parameter_gradient(
    theta: torch.Tensor,
    k_min: int,
    k_max: int,
    temperature: float = 1.0
) -> Tuple[int, torch.Tensor]:
    """
    Calculate gradient for discrete parameter optimization.
    
    Implements: k = floor(σ(θ) * (k_max - k_min) + 0.5)
    
    Args:
        theta: Continuous parameter
        k_min: Minimum discrete value
        k_max: Maximum discrete value
        temperature: Temperature for sigmoid
        
    Returns:
        Discrete value and gradient tensor
    """
    # Apply temperature-scaled sigmoid
    sigma = torch.sigmoid(theta / temperature)
    
    # Map to discrete range
    k_continuous = sigma * (k_max - k_min) + k_min
    k_discrete = int(torch.floor(k_continuous + 0.5).item())
    
    # Gradient flows through continuous relaxation
    return k_discrete, k_continuous


def calculate_information_gain(
    old_importance: Dict[str, float],
    new_importance: Dict[str, float]
) -> float:
    """
    Calculate information gain between two network states.
    
    Args:
        old_importance: Layer importances before modification
        new_importance: Layer importances after modification
        
    Returns:
        Information gain (can be negative for information loss)
    """
    # Get common layers
    common_layers = set(old_importance.keys()) & set(new_importance.keys())
    
    if not common_layers:
        return 0.0
    
    # Calculate gain for common layers
    gain = 0.0
    for layer in common_layers:
        gain += new_importance[layer] - old_importance[layer]
    
    # Penalize removed layers
    removed_layers = set(old_importance.keys()) - set(new_importance.keys())
    for layer in removed_layers:
        gain -= old_importance[layer]
    
    # Reward new layers (conservative estimate)
    new_layers = set(new_importance.keys()) - set(old_importance.keys())
    if new_layers and common_layers:
        avg_importance = float(np.mean([new_importance[l] for l in common_layers]))
        gain += len(new_layers) * avg_importance * 0.5  # Conservative factor
    
    return gain


def calculate_task_complexity(
    num_classes: int,
    input_dim: int,
    dataset_size: int,
    task_type: str = 'classification'
) -> float:
    """
    Estimate task complexity for adaptive thresholding.
    
    Args:
        num_classes: Number of output classes
        input_dim: Input dimensionality
        dataset_size: Number of training samples
        task_type: Type of task
        
    Returns:
        Complexity score between 0 and 1
    """
    # Base complexity factors
    class_complexity = np.log2(num_classes + 1) / 10  # Normalize by 1024 classes
    dim_complexity = np.log10(input_dim + 1) / 5      # Normalize by 100k dims
    size_complexity = np.log10(dataset_size + 1) / 6   # Normalize by 1M samples
    
    # Task-specific weights
    task_weights = {
        'classification': [0.4, 0.3, 0.3],
        'regression': [0.2, 0.5, 0.3],
        'generation': [0.3, 0.4, 0.3],
        'detection': [0.35, 0.35, 0.3]
    }
    
    weights = task_weights.get(task_type, [0.33, 0.33, 0.34])
    
    # Weighted combination
    complexity = (
        weights[0] * class_complexity +
        weights[1] * dim_complexity +
        weights[2] * size_complexity
    )
    
    return max(0, min(1, complexity)) 