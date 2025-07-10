"""
Adaptive Entropy Control module for Neuro Exapt.

This module implements adaptive entropy threshold management for guiding
structural evolution decisions in neural networks.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import json
import os


@dataclass
class EntropyMetrics:
    """Container for entropy-related metrics."""
    current_entropy: float
    threshold: float
    layer_entropies: Dict[str, float]
    average_entropy: float
    min_entropy: float
    max_entropy: float
    epoch: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "current_entropy": self.current_entropy,
            "threshold": self.threshold,
            "layer_entropies": self.layer_entropies,
            "average_entropy": self.average_entropy,
            "min_entropy": self.min_entropy,
            "max_entropy": self.max_entropy,
            "epoch": self.epoch
        }


class AdaptiveEntropy:
    """
    Adaptive Entropy Controller for dynamic threshold adjustment.
    
    Implements adaptive entropy thresholding based on:
    τ = τ₀ · exp(-γ · Epoch) · (1 + δ · TaskComplexity)
    """
    
    def __init__(
        self,
        initial_threshold: float = 0.5,
        decay_rate: float = 0.05,
        min_threshold: float = 0.1,
        task_complexity_factor: float = 0.2,
        window_size: int = 10,
        adapt_rate: float = 0.1
    ):
        """
        Initialize Adaptive Entropy Controller.
        
        Args:
            initial_threshold: Starting entropy threshold τ₀
            decay_rate: Entropy decay rate γ
            min_threshold: Minimum allowable threshold
            task_complexity_factor: Task complexity coefficient δ
            window_size: Size of moving average window
            adapt_rate: Rate of threshold adaptation
        """
        self.initial_threshold = initial_threshold
        self.threshold = initial_threshold
        self.decay_rate = decay_rate
        self.min_threshold = min_threshold
        self.task_complexity_factor = task_complexity_factor
        self.window_size = window_size
        self.adapt_rate = adapt_rate
        
        # History tracking
        self.entropy_history: List[float] = []
        self.threshold_history: List[float] = []
        self.metrics_history: List[EntropyMetrics] = []
        
        # Task complexity estimation
        self.estimated_complexity = 1.0
        self.complexity_estimator: Optional[Callable] = None
        
        # State
        self.current_epoch = 0
        self._layer_entropy_cache: Dict[str, float] = {}
        
    def update_threshold(
        self,
        epoch: int,
        task_complexity: Optional[float] = None
    ) -> float:
        """
        Update entropy threshold based on current epoch and task complexity.
        
        Implements: τ = τ₀ · exp(-γ · Epoch) · (1 + δ · TaskComplexity)
        
        Args:
            epoch: Current training epoch
            task_complexity: Optional explicit task complexity (0-1)
            
        Returns:
            Updated threshold value
        """
        self.current_epoch = epoch
        
        # Use provided complexity or estimate
        if task_complexity is not None:
            self.estimated_complexity = task_complexity
        elif self.complexity_estimator is not None:
            self.estimated_complexity = self.complexity_estimator(epoch)
        
        # Calculate adaptive threshold
        decay_factor = np.exp(-self.decay_rate * epoch)
        complexity_factor = 1 + self.task_complexity_factor * self.estimated_complexity
        
        self.threshold = self.initial_threshold * decay_factor * complexity_factor
        self.threshold = max(self.min_threshold, self.threshold)
        
        # Record history
        self.threshold_history.append(self.threshold)
        
        return self.threshold
    
    def measure(
        self,
        outputs: torch.Tensor,
        layer_outputs: Optional[Dict[str, torch.Tensor]] = None
    ) -> float:
        """
        Measure current entropy from model outputs.
        
        Args:
            outputs: Model outputs
            layer_outputs: Optional dict of layer outputs
            
        Returns:
            Current entropy value
        """
        # Calculate output entropy
        if outputs.dim() > 1 and outputs.size(1) > 1:
            # Multi-class output - use softmax entropy
            probs = torch.softmax(outputs, dim=-1)
            log_probs = torch.log(probs + 1e-10)
            entropy = -torch.sum(probs * log_probs, dim=-1).mean()
            current_entropy = entropy.item()
        else:
            # Single output - use variance as proxy for entropy
            current_entropy = outputs.var().item()
        
        # Calculate layer entropies if provided
        if layer_outputs:
            self._layer_entropy_cache.clear()
            for name, layer_out in layer_outputs.items():
                if layer_out.dim() > 1:
                    layer_probs = torch.softmax(layer_out.view(layer_out.size(0), -1), dim=-1)
                    layer_log_probs = torch.log(layer_probs + 1e-10)
                    layer_entropy = -torch.sum(layer_probs * layer_log_probs, dim=-1).mean()
                    self._layer_entropy_cache[name] = layer_entropy.item()
                else:
                    self._layer_entropy_cache[name] = layer_out.var().item()
        
        # Update history
        self.entropy_history.append(current_entropy)
        if len(self.entropy_history) > self.window_size * 10:
            self.entropy_history = self.entropy_history[-self.window_size * 10:]
        
        return current_entropy
    
    def should_prune(self, current_entropy: Optional[float] = None) -> bool:
        """
        Determine if pruning should occur based on entropy threshold.
        
        Args:
            current_entropy: Current entropy value (uses last measured if None)
            
        Returns:
            True if entropy is below threshold and pruning should occur
        """
        if current_entropy is None:
            if not self.entropy_history:
                return False
            current_entropy = self.entropy_history[-1]
        
        return current_entropy < self.threshold
    
    def should_expand(self, current_entropy: Optional[float] = None) -> bool:
        """
        Determine if expansion should occur based on entropy analysis.
        
        Args:
            current_entropy: Current entropy value
            
        Returns:
            True if expansion is recommended
        """
        if current_entropy is None:
            if not self.entropy_history:
                return False
            current_entropy = self.entropy_history[-1]
        
        # Expand if entropy is significantly above threshold
        # or if entropy has been increasing
        expand_threshold = self.threshold * 1.5
        
        if current_entropy > expand_threshold:
            return True
        
        # Check if entropy is increasing (potential capacity limit)
        if len(self.entropy_history) >= self.window_size:
            recent_entropy = self.entropy_history[-self.window_size:]
            entropy_trend = np.polyfit(range(len(recent_entropy)), recent_entropy, 1)[0]
            if entropy_trend > 0.01:  # Positive trend
                return True
        
        return False
    
    def get_moving_average(self, window: Optional[int] = None) -> float:
        """
        Get moving average of entropy values.
        
        Args:
            window: Window size (uses instance default if None)
            
        Returns:
            Moving average entropy
        """
        window = window or self.window_size
        if len(self.entropy_history) < window:
            return float(np.mean(self.entropy_history)) if self.entropy_history else 0.0
        
        return float(np.mean(self.entropy_history[-window:]))
    
    def adapt_threshold(self, performance_metric: float, target_metric: float):
        """
        Adapt threshold based on performance feedback.
        
        Args:
            performance_metric: Current performance (e.g., accuracy)
            target_metric: Target performance
        """
        # If performance is below target, increase threshold (less pruning)
        # If performance is above target, decrease threshold (more pruning)
        performance_gap = target_metric - performance_metric
        
        adjustment = self.adapt_rate * performance_gap
        self.threshold *= (1 + adjustment)
        self.threshold = max(self.min_threshold, self.threshold)
    
    def estimate_task_complexity(
        self,
        dataset_size: int,
        num_classes: int,
        input_dim: int,
        model_depth: int
    ) -> float:
        """
        Estimate task complexity based on dataset and model characteristics.
        
        Args:
            dataset_size: Number of training samples
            num_classes: Number of output classes
            input_dim: Input dimensionality
            model_depth: Model depth
            
        Returns:
            Estimated complexity (0-1)
        """
        # Normalize factors
        size_factor = np.log10(dataset_size + 1) / 6  # Assume max 1M samples
        class_factor = np.log2(num_classes + 1) / 10  # Assume max 1024 classes
        dim_factor = np.log10(input_dim + 1) / 5      # Assume max 100k dimensions
        depth_factor = model_depth / 100               # Assume max depth 100
        
        # Weighted combination
        complexity = (
            0.3 * size_factor +
            0.3 * class_factor +
            0.2 * dim_factor +
            0.2 * depth_factor
        )
        
        # Clamp to [0, 1]
        self.estimated_complexity = max(0, min(1, complexity))
        
        return self.estimated_complexity
    
    def set_complexity_estimator(self, estimator: Callable[[int], float]):
        """
        Set custom complexity estimator function.
        
        Args:
            estimator: Function that takes epoch and returns complexity
        """
        self.complexity_estimator = estimator
    
    def get_layer_entropies(self) -> Dict[str, float]:
        """Get cached layer entropy values."""
        return self._layer_entropy_cache.copy()
    
    def get_metrics(self) -> EntropyMetrics:
        """
        Get current entropy metrics.
        
        Returns:
            EntropyMetrics object with current state
        """
        current_entropy = self.entropy_history[-1] if self.entropy_history else 0.0
        layer_entropies = self.get_layer_entropies()
        
        if layer_entropies:
            avg_entropy = float(np.mean(list(layer_entropies.values())))
            min_entropy = min(layer_entropies.values())
            max_entropy = max(layer_entropies.values())
        else:
            avg_entropy = min_entropy = max_entropy = current_entropy
        
        metrics = EntropyMetrics(
            current_entropy=current_entropy,
            threshold=self.threshold,
            layer_entropies=layer_entropies,
            average_entropy=avg_entropy,
            min_entropy=min_entropy,
            max_entropy=max_entropy,
            epoch=self.current_epoch
        )
        
        self.metrics_history.append(metrics)
        
        return metrics
    
    def save_state(self, filepath: str):
        """
        Save controller state to file.
        
        Args:
            filepath: Path to save file
        """
        state = {
            "threshold": self.threshold,
            "current_epoch": self.current_epoch,
            "estimated_complexity": self.estimated_complexity,
            "entropy_history": self.entropy_history[-self.window_size * 10:],
            "threshold_history": self.threshold_history,
            "parameters": {
                "initial_threshold": self.initial_threshold,
                "decay_rate": self.decay_rate,
                "min_threshold": self.min_threshold,
                "task_complexity_factor": self.task_complexity_factor,
                "window_size": self.window_size,
                "adapt_rate": self.adapt_rate
            }
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, filepath: str):
        """
        Load controller state from file.
        
        Args:
            filepath: Path to saved state file
        """
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.threshold = state["threshold"]
        self.current_epoch = state["current_epoch"]
        self.estimated_complexity = state["estimated_complexity"]
        self.entropy_history = state["entropy_history"]
        self.threshold_history = state["threshold_history"]
        
        # Update parameters if different
        params = state.get("parameters", {})
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def plot_history(self, save_path: Optional[str] = None):
        """
        Plot entropy and threshold history.
        
        Args:
            save_path: Optional path to save plot
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Plot entropy history
            if self.entropy_history:
                ax1.plot(self.entropy_history, label='Entropy', alpha=0.7)
                if len(self.entropy_history) >= self.window_size:
                    ma = [self.get_moving_average() for _ in range(len(self.entropy_history))]
                    ax1.plot(ma, label=f'MA({self.window_size})', linewidth=2)
            
            ax1.axhline(y=self.threshold, color='r', linestyle='--', label='Current Threshold')
            ax1.set_xlabel('Step')
            ax1.set_ylabel('Entropy')
            ax1.set_title('Entropy Evolution')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot threshold history
            if self.threshold_history:
                ax2.plot(self.threshold_history, label='Threshold', color='red')
                ax2.axhline(y=self.min_threshold, color='gray', linestyle=':', 
                           label='Min Threshold')
            
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Threshold')
            ax2.set_title('Adaptive Threshold Evolution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
            else:
                plt.show()
                
            plt.close()
            
        except ImportError:
            print("Matplotlib not available for plotting") 