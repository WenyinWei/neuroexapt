"""
Optimization module for Neuro Exapt.

This module implements optimization algorithms and utilities for
information-theoretic neural network training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
from abc import ABC, abstractmethod


class DiscreteParameterOptimizer:
    """
    Optimizer for discrete parameters using continuous relaxation.
    
    Implements differentiable optimization of discrete architectural choices.
    """
    
    def __init__(
        self,
        parameters: List[torch.nn.Parameter],
        ranges: Dict[str, Tuple[int, int]],
        lr: float = 0.01,
        temperature: float = 1.0,
        temperature_decay: float = 0.95,
        min_temperature: float = 0.1
    ):
        """
        Initialize discrete parameter optimizer.
        
        Args:
            parameters: List of continuous parameters to optimize
            ranges: Mapping from parameter names to (min, max) ranges
            lr: Learning rate
            temperature: Initial temperature for Gumbel-softmax
            temperature_decay: Temperature decay rate
            min_temperature: Minimum temperature
        """
        self.parameters = parameters
        self.ranges = ranges
        self.lr = lr
        self.temperature = temperature
        self.temperature_decay = temperature_decay
        self.min_temperature = min_temperature
        
        # Initialize continuous parameters
        self.continuous_params = {}
        for i, param in enumerate(parameters):
            # Initialize near middle of range
            param.data.normal_(0, 0.1)
            self.continuous_params[i] = param
            
    def step(self, gradients: Optional[Dict[int, torch.Tensor]] = None):
        """
        Perform optimization step.
        
        Args:
            gradients: Optional manual gradients
        """
        with torch.no_grad():
            for idx, param in self.continuous_params.items():
                if gradients and idx in gradients:
                    grad = gradients[idx]
                else:
                    grad = param.grad
                    
                if grad is not None:
                    # Gradient descent on continuous parameters
                    param.data -= self.lr * grad
                    
    def get_discrete_values(self) -> Dict[int, int]:
        """
        Get current discrete parameter values.
        
        Returns:
            Mapping from parameter index to discrete value
        """
        discrete_values = {}
        
        for idx, param in self.continuous_params.items():
            # Apply sigmoid
            sigma = torch.sigmoid(param / self.temperature)
            
            # Map to discrete range (assuming all params use same range for simplicity)
            # In practice, you'd look up the specific range for each parameter
            k_min, k_max = 1, 7  # Default range
            k_continuous = sigma * (k_max - k_min) + k_min
            k_discrete = int(torch.floor(k_continuous + 0.5).item())
            
            discrete_values[idx] = k_discrete
            
        return discrete_values
        
    def anneal_temperature(self):
        """Anneal temperature for better discrete convergence."""
        self.temperature = max(
            self.min_temperature,
            self.temperature * self.temperature_decay
        )


class InformationBottleneckOptimizer(optim.Optimizer):
    """
    Custom optimizer that incorporates information bottleneck principles.
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        beta: float = 0.999,
        ib_weight: float = 0.1,
        compression_target: Optional[float] = None
    ):
        """
        Initialize IB-aware optimizer.
        
        Args:
            params: Model parameters
            lr: Learning rate
            beta: Momentum factor
            ib_weight: Weight for information bottleneck term
            compression_target: Target compression ratio
        """
        defaults = dict(
            lr=lr,
            beta=beta,
            ib_weight=ib_weight,
            compression_target=compression_target
        )
        super().__init__(params, defaults)
        
    def step(self, closure: Optional[Callable] = None):
        """Perform optimization step with IB regularization."""
        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('IB optimizer does not support sparse gradients')
                    
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['compression_penalty'] = 0.0
                    
                exp_avg = state['exp_avg']
                beta = group['beta']
                state['step'] += 1
                
                # Momentum update
                exp_avg.mul_(beta).add_(grad, alpha=1 - beta)
                
                # Information bottleneck regularization
                if group['ib_weight'] > 0:
                    # Add compression penalty based on parameter magnitude
                    compression = torch.norm(p.data, p=2) / (p.numel() ** 0.5)
                    ib_penalty = group['ib_weight'] * compression
                    
                    # Track compression
                    state['compression_penalty'] = ib_penalty.item()
                    
                    # Modify gradient
                    grad_with_ib = exp_avg + ib_penalty * p.data
                else:
                    grad_with_ib = exp_avg
                    
                # Parameter update
                p.data.add_(grad_with_ib, alpha=-group['lr'])
                
        return loss


class StructuralGradientEstimator:
    """
    Estimates gradients for structural modifications using finite differences
    or REINFORCE-style estimation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable,
        method: str = 'finite_diff',
        epsilon: float = 0.01
    ):
        """
        Initialize gradient estimator.
        
        Args:
            model: Neural network model
            loss_fn: Loss function
            method: Gradient estimation method
            epsilon: Perturbation size for finite differences
        """
        self.model = model
        self.loss_fn = loss_fn
        self.method = method
        self.epsilon = epsilon
        
    def estimate_pruning_gradient(
        self,
        layer_name: str,
        data_batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> float:
        """
        Estimate gradient of loss w.r.t. pruning a layer.
        
        Args:
            layer_name: Name of layer to evaluate
            data_batch: Batch of (inputs, targets)
            
        Returns:
            Estimated gradient
        """
        inputs, targets = data_batch
        
        # Original loss
        with torch.no_grad():
            outputs_orig = self.model(inputs)
            loss_orig = self.loss_fn(outputs_orig, targets).item()
            
        # Temporarily prune layer
        layer = dict(self.model.named_modules())[layer_name]
        original_forward = layer.forward
        
        def identity_forward(x):
            return x if hasattr(layer, 'in_features') else x
            
        layer.forward = identity_forward
        
        # Loss after pruning
        with torch.no_grad():
            outputs_pruned = self.model(inputs)
            loss_pruned = self.loss_fn(outputs_pruned, targets).item()
            
        # Restore original forward
        layer.forward = original_forward
        
        # Gradient estimate
        gradient = (loss_pruned - loss_orig) / self.epsilon
        
        return gradient
        
    def estimate_expansion_gradient(
        self,
        layer_name: str,
        expansion_factor: float,
        data_batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> float:
        """
        Estimate gradient of loss w.r.t. expanding a layer.
        
        This is more complex as it requires actually modifying the layer.
        Simplified implementation provided.
        """
        # In practice, this would create an expanded version of the layer
        # and estimate the gradient. For now, return a heuristic.
        
        inputs, targets = data_batch
        
        with torch.no_grad():
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets).item()
            
        # Heuristic: expansion helps if loss is high
        gradient = -loss * expansion_factor
        
        return gradient


class AdaptiveLearningRateScheduler:
    """
    Learning rate scheduler that adapts based on information-theoretic metrics.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_lr: float,
        entropy_weight: float = 0.1,
        redundancy_weight: float = 0.1,
        min_lr: float = 1e-6,
        max_lr: float = 1.0
    ):
        """
        Initialize adaptive scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            initial_lr: Initial learning rate
            entropy_weight: Weight for entropy-based adjustment
            redundancy_weight: Weight for redundancy-based adjustment
            min_lr: Minimum learning rate
            max_lr: Maximum learning rate
        """
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.entropy_weight = entropy_weight
        self.redundancy_weight = redundancy_weight
        self.min_lr = min_lr
        self.max_lr = max_lr
        
        self.current_lr = initial_lr
        self.lr_history = [initial_lr]
        
    def step(
        self,
        entropy: float,
        redundancy: float,
        performance_metric: Optional[float] = None
    ):
        """
        Update learning rate based on metrics.
        
        Args:
            entropy: Current network entropy
            redundancy: Current redundancy score
            performance_metric: Optional performance metric (e.g., accuracy)
        """
        # Base adjustment from entropy
        # High entropy -> more exploration -> higher LR
        entropy_factor = 1 + self.entropy_weight * (entropy - 0.5)
        
        # Redundancy adjustment
        # High redundancy -> need structural changes -> higher LR
        redundancy_factor = 1 + self.redundancy_weight * redundancy
        
        # Performance-based adjustment if available
        if performance_metric is not None:
            # Slow down if performing well
            perf_factor = 2 - performance_metric  # Assumes metric in [0, 1]
        else:
            perf_factor = 1.0
            
        # Combined adjustment
        lr_scale = entropy_factor * redundancy_factor * perf_factor
        
        # Update learning rate
        new_lr = self.initial_lr * lr_scale
        new_lr = max(self.min_lr, min(self.max_lr, new_lr))
        
        self.current_lr = new_lr
        self.lr_history.append(new_lr)
        
        # Update optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
            
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.current_lr


def create_optimization_schedule(
    total_epochs: int,
    warmup_epochs: int = 5,
    structure_update_frequency: int = 10,
    final_structure_epoch: Optional[int] = None
) -> Dict[str, Any]:
    """
    Create optimization schedule for Neuro Exapt training.
    
    Args:
        total_epochs: Total number of training epochs
        warmup_epochs: Number of warmup epochs
        structure_update_frequency: How often to update structure
        final_structure_epoch: Last epoch for structural changes
        
    Returns:
        Schedule dictionary
    """
    if final_structure_epoch is None:
        final_structure_epoch = int(0.8 * total_epochs)
        
    schedule = {
        'total_epochs': total_epochs,
        'warmup_epochs': warmup_epochs,
        'structure_update_frequency': structure_update_frequency,
        'final_structure_epoch': final_structure_epoch,
        'phases': []
    }
    
    # Warmup phase
    schedule['phases'].append({
        'name': 'warmup',
        'start': 0,
        'end': warmup_epochs,
        'lr_scale': lambda epoch: epoch / warmup_epochs,
        'allow_pruning': False,
        'allow_expansion': False,
        'allow_mutation': False
    })
    
    # Main training phase
    schedule['phases'].append({
        'name': 'main',
        'start': warmup_epochs,
        'end': final_structure_epoch,
        'lr_scale': lambda epoch: 1.0,
        'allow_pruning': True,
        'allow_expansion': True,
        'allow_mutation': True
    })
    
    # Fine-tuning phase
    schedule['phases'].append({
        'name': 'fine_tuning',
        'start': final_structure_epoch,
        'end': total_epochs,
        'lr_scale': lambda epoch: 0.5 * (1 + np.cos(np.pi * (epoch - final_structure_epoch) / (total_epochs - final_structure_epoch))),
        'allow_pruning': False,
        'allow_expansion': False,
        'allow_mutation': False
    })
    
    return schedule


def calculate_gradient_information(
    gradients: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """
    Calculate information-theoretic metrics from gradients.
    
    Args:
        gradients: Dictionary of gradients by layer name
        
    Returns:
        Dictionary of gradient metrics
    """
    metrics = {}
    
    all_grads = []
    for name, grad in gradients.items():
        if grad is not None:
            flat_grad = grad.view(-1)
            all_grads.append(flat_grad)
            
            # Per-layer metrics
            metrics[f'{name}_norm'] = torch.norm(grad).item()
            metrics[f'{name}_mean'] = torch.mean(torch.abs(grad)).item()
            metrics[f'{name}_std'] = torch.std(grad).item()
            
    if all_grads:
        # Global metrics
        all_grads = torch.cat(all_grads)
        
        # Gradient entropy (discretized)
        hist = torch.histc(all_grads, bins=50)
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        gradient_entropy = -torch.sum(hist * torch.log2(hist)).item()
        
        metrics['gradient_entropy'] = gradient_entropy
        metrics['gradient_norm'] = torch.norm(all_grads).item()
        metrics['gradient_sparsity'] = (torch.abs(all_grads) < 1e-6).float().mean().item()
        
    return metrics 