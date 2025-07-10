"""
Information Theory module for Neuro Exapt.

This module implements the Information Bottleneck principle and related
information-theoretic measures for neural network optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from scipy.stats import entropy as scipy_entropy


class InformationBottleneck:
    """
    Information Bottleneck engine for measuring and optimizing information flow
    through neural network layers.
    
    Based on the principle: I(X;Y) - β·I(Y;Z)
    where X is input, Y is layer representation, Z is output.
    """
    
    def __init__(
        self,
        beta: float = 1.0,
        n_bins: int = 30,
        normalize: bool = True,
        device: Optional[torch.device] = None
    ):
        """
        Initialize Information Bottleneck engine.
        
        Args:
            beta: Trade-off parameter between compression and prediction
            n_bins: Number of bins for entropy estimation
            normalize: Whether to normalize mutual information values
            device: Torch device for computations
        """
        self.beta = beta
        self.n_bins = n_bins
        self.normalize = normalize
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Cache for layer activations
        self._activation_cache: Dict[str, torch.Tensor] = {}
        self._hooks: List = []
        
    def calculate_entropy(self, tensor: torch.Tensor, dim: Optional[int] = None) -> float:
        """
        Calculate Shannon entropy of a tensor.
        
        Args:
            tensor: Input tensor
            dim: Dimension along which to calculate entropy
            
        Returns:
            Entropy value
        """
        if dim is not None:
            # Calculate entropy along specific dimension
            probs = F.softmax(tensor, dim=dim)
            log_probs = torch.log(probs + 1e-10)
            entropy = -torch.sum(probs * log_probs, dim=dim)
            return entropy.mean().item()
        
        # Flatten tensor and estimate entropy using histogram
        data = tensor.detach().cpu().numpy().flatten()
        
        # Create histogram
        hist, _ = np.histogram(data, bins=self.n_bins)
        hist = hist + 1e-10  # Avoid log(0)
        hist = hist / hist.sum()
        
        return float(scipy_entropy(hist))
    
    def calculate_mutual_information(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        normalize: Optional[bool] = None
    ) -> float:
        """
        Calculate mutual information I(X;Y) using binning method.
        
        Args:
            x: First variable
            y: Second variable
            normalize: Whether to normalize (overrides instance setting)
            
        Returns:
            Mutual information value
        """
        normalize = self.normalize if normalize is None else normalize
        
        # Ensure tensors are on CPU for numpy operations
        x_np = x.detach().cpu().numpy().flatten()
        y_np = y.detach().cpu().numpy().flatten()
        
        # Create 2D histogram
        hist_2d, _, _ = np.histogram2d(x_np, y_np, bins=self.n_bins)
        hist_2d = hist_2d + 1e-10
        
        # Normalize to get joint probability
        p_xy = hist_2d / hist_2d.sum()
        
        # Marginal distributions
        p_x = p_xy.sum(axis=1)
        p_y = p_xy.sum(axis=0)
        
        # Calculate MI: I(X;Y) = H(X) + H(Y) - H(X,Y)
        h_x = float(scipy_entropy(p_x))
        h_y = float(scipy_entropy(p_y))
        h_xy = float(scipy_entropy(p_xy.flatten()))
        
        mi = h_x + h_y - h_xy
        
        if normalize:
            # Normalize by min(H(X), H(Y))
            mi = mi / (min(h_x, h_y) + 1e-10)
            
        return float(mi)
    
    def calculate_layer_importance(
        self,
        layer_output: torch.Tensor,
        target_output: torch.Tensor,
        task_type: str = "classification",
        layer_name: Optional[str] = None
    ) -> float:
        """
        Calculate layer importance I(L_i;O) with task-aware weighting.
        
        Implements: I(L_i;O) = H(O) - H(O|L_i) · ψ(TaskType)
        
        Args:
            layer_output: Output of layer i
            target_output: Target/final output
            task_type: Type of task (classification, generation, regression)
            layer_name: Optional layer identifier for caching
            
        Returns:
            Layer importance score
        """
        # Task-specific weight function ψ
        task_weights = {
            "classification": 1.2,
            "generation": 0.8,
            "regression": 1.0,
            "detection": 1.1
        }
        psi = task_weights.get(task_type, 1.0)
        
        # Calculate H(O)
        h_output = self.calculate_entropy(target_output)
        
        # Estimate H(O|L_i) using conditional entropy approximation
        # This is approximated as H(O) - I(L_i;O)
        mi = self.calculate_mutual_information(layer_output, target_output)
        h_output_given_layer = h_output - mi
        
        # Apply task-aware weighting
        importance = h_output - h_output_given_layer * psi
        
        # Cache result if layer name provided
        if layer_name:
            self._activation_cache[f"{layer_name}_importance"] = importance
            
        return importance
    
    def calculate_redundancy(
        self,
        layer_importances: List[float],
        output_entropy: float,
        depth: int,
        depth_decay_lambda: float = 0.03
    ) -> float:
        """
        Calculate network redundancy.
        
        Implements: R = 1 - Σ I(L_i;O) / (H(O) · exp(-λ · Depth))
        
        Args:
            layer_importances: List of layer importance values
            output_entropy: Entropy of output H(O)
            depth: Network depth
            depth_decay_lambda: Decay factor for depth normalization
            
        Returns:
            Redundancy score (0 to 1)
        """
        # Sum of layer importances
        total_importance = sum(layer_importances)
        
        # Depth-aware normalization
        normalization = output_entropy * np.exp(-depth_decay_lambda * depth)
        
        # Calculate redundancy
        redundancy = 1 - (total_importance / (normalization + 1e-10))
        
        return max(0, min(1, redundancy))  # Clamp to [0, 1]
    
    def register_hooks(self, model: nn.Module, layer_names: Optional[List[str]] = None):
        """
        Register forward hooks to capture layer activations.
        
        Args:
            model: PyTorch model
            layer_names: Specific layers to monitor (None for all)
        """
        self.clear_hooks()
        
        def hook_fn(name):
            def hook(module, input, output):
                self._activation_cache[name] = output.detach()
            return hook
        
        # Register hooks for specified layers or all layers
        for name, module in model.named_modules():
            if layer_names is None or name in layer_names:
                if len(list(module.children())) == 0:  # Leaf module
                    handle = module.register_forward_hook(hook_fn(name))
                    self._hooks.append(handle)
    
    def clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._activation_cache.clear()
    
    def compute_information_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        layer_outputs: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Compute information-theoretic loss for training.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            layer_outputs: Optional dict of layer outputs
            
        Returns:
            Information loss tensor
        """
        # Use cached activations if layer_outputs not provided
        if layer_outputs is None:
            layer_outputs = self._activation_cache
        
        # Base prediction loss (cross-entropy for classification)
        if predictions.dim() > 1 and predictions.size(1) > 1:
            # Multi-class classification
            base_loss = F.cross_entropy(predictions, targets)
        else:
            # Binary classification or regression
            base_loss = F.mse_loss(predictions, targets.float())
        
        # Information bottleneck regularization
        ib_loss = 0.0
        if layer_outputs:
            for name, output in layer_outputs.items():
                if "importance" not in name:  # Skip cached importance values
                    # Minimize I(layer;input) while preserving I(layer;output)
                    layer_entropy = self.calculate_entropy(output)
                    ib_loss += self.beta * layer_entropy
        
        total_loss = base_loss + ib_loss
        
        return total_loss
    
    def analyze_network(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        task_type: str = "classification"
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Comprehensive information-theoretic analysis of a neural network.
        
        Args:
            model: Neural network model
            dataloader: Data loader for analysis
            task_type: Type of task
            
        Returns:
            Dictionary containing analysis results
        """
        model.eval()
        self.register_hooks(model)
        
        layer_importances = {}
        layer_entropies = {}
        total_mi = 0.0
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                if batch_idx > 10:  # Analyze first 10 batches
                    break
                    
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                
                # Analyze each layer
                for name, activation in self._activation_cache.items():
                    if "importance" not in name:
                        # Calculate layer entropy
                        entropy = self.calculate_entropy(activation)
                        layer_entropies[name] = layer_entropies.get(name, 0) + entropy
                        
                        # Calculate layer importance
                        importance = self.calculate_layer_importance(
                            activation, outputs, task_type, name
                        )
                        layer_importances[name] = layer_importances.get(name, 0) + importance
                        
                        # Calculate mutual information with input
                        mi = self.calculate_mutual_information(
                            inputs.view(inputs.size(0), -1),
                            activation.view(activation.size(0), -1)
                        )
                        total_mi += mi
        
        # Average over batches
        n_batches = min(len(dataloader), 10)
        layer_importances = {k: v/n_batches for k, v in layer_importances.items()}
        layer_entropies = {k: v/n_batches for k, v in layer_entropies.items()}
        
        # Calculate overall metrics
        output_entropy = self.calculate_entropy(outputs)
        redundancy = self.calculate_redundancy(
            list(layer_importances.values()),
            output_entropy,
            len(layer_importances)
        )
        
        self.clear_hooks()
        
        return {
            "layer_importances": layer_importances,
            "layer_entropies": layer_entropies,
            "output_entropy": output_entropy,
            "redundancy": redundancy,
            "average_mutual_information": total_mi / (n_batches * len(layer_importances))
        }


class AdaptiveInformationBottleneck(InformationBottleneck):
    """
    Extended Information Bottleneck with adaptive beta scheduling.
    """
    
    def __init__(
        self,
        initial_beta: float = 1.0,
        beta_schedule: str = "linear",
        min_beta: float = 0.1,
        max_beta: float = 10.0,
        **kwargs
    ):
        """
        Initialize adaptive information bottleneck.
        
        Args:
            initial_beta: Starting beta value
            beta_schedule: Schedule type (linear, exponential, cosine)
            min_beta: Minimum beta value
            max_beta: Maximum beta value
            **kwargs: Additional arguments for parent class
        """
        super().__init__(beta=initial_beta, **kwargs)
        self.initial_beta = initial_beta
        self.beta_schedule = beta_schedule
        self.min_beta = min_beta
        self.max_beta = max_beta
        self.current_epoch = 0
        
    def update_beta(self, epoch: int, total_epochs: int):
        """Update beta according to schedule."""
        self.current_epoch = epoch
        progress = epoch / total_epochs
        
        if self.beta_schedule == "linear":
            self.beta = self.initial_beta + (self.max_beta - self.initial_beta) * progress
        elif self.beta_schedule == "exponential":
            self.beta = self.initial_beta * (self.max_beta / self.initial_beta) ** progress
        elif self.beta_schedule == "cosine":
            self.beta = self.min_beta + 0.5 * (self.max_beta - self.min_beta) * \
                       (1 + np.cos(np.pi * (1 - progress)))
        else:
            self.beta = self.initial_beta
            
        self.beta = max(self.min_beta, min(self.max_beta, self.beta)) 