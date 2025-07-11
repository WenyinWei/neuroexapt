"""
Main NeuroExapt class that integrates all components.

This is the primary interface for using the Neuro Exapt framework.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
import warnings
import os
import yaml
from datetime import datetime
import torch.nn.functional as F

from .core import (
    InformationBottleneck,
    AdaptiveEntropy,
    StructuralEvolution,
    PruneByEntropy,
    ExpandWithMI,
    MutateDiscrete,
    CompoundOperator
)
from .math import (
    calculate_network_complexity,
    calculate_task_complexity,
    calculate_redundancy_score
)
from .utils.logging import setup_logger, log_metrics
from .utils.visualization import plot_evolution_history


class NeuroExapt:
    """
    Main class for Neuro Exapt framework.
    
    Integrates information bottleneck, adaptive entropy control,
    and structural evolution for dynamic neural architecture optimization.
    """
    
    def __init__(
        self,
        task_type: str = "classification",
        depth: Optional[int] = None,
        entropy_weight: float = 0.5,
        info_weight: float = 0.5,
        config_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        verbose: bool = True
    ):
        """
        Initialize Neuro Exapt.
        
        Args:
            task_type: Type of task (classification, regression, generation)
            depth: Initial network depth (optional)
            entropy_weight: Weight for entropy-based decisions
            info_weight: Weight for information-theoretic loss
            config_path: Path to configuration file
            device: Torch device for computation
            verbose: Whether to print progress
        """
        self.task_type = task_type
        self.depth = depth
        self.entropy_weight = entropy_weight
        self.info_weight = info_weight
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.info_theory = InformationBottleneck(
            beta=self.config['information']['bottleneck_weight'],
            device=self.device
        )
        
        self.entropy_ctrl = AdaptiveEntropy(
            initial_threshold=self.config['entropy']['initial_threshold'],
            decay_rate=self.config['entropy']['decay_rate'],
            min_threshold=self.config['entropy']['min_threshold'],
            task_complexity_factor=self.config['entropy']['task_complexity_factor']
        )
        
        self.struct_optimizer = StructuralEvolution(
            alpha=self.config['training']['info_retention_alpha'],
            beta=self.config['training']['structure_variation_beta'],
            prune_ratio=self.config['evolution']['prune_threshold'],
            min_layers=self.config['evolution']['min_layers'],
            max_layers=self.config['evolution']['max_layers'],
            device=self.device
        )
        
        # Initialize operators
        self._init_operators()
        
        # State tracking
        self.model = None
        self.wrapped_model = None
        self.current_epoch = 0
        self.metrics_history = []
        self.evolution_history = []
        
        # Setup logging
        if self.verbose:
            self.logger = setup_logger("NeuroExapt")
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        # Default configuration
        default_config_path = os.path.join(
            os.path.dirname(__file__),
            'config.yaml'
        )
        
        config_path = config_path or default_config_path
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            # Fallback to hardcoded defaults
            config = {
                'information': {
                    'bottleneck_weight': 0.5,
                    'layer_importance_weight': 1.0,
                    'task_type_weights': {
                        'classification': 1.2,
                        'generation': 0.8,
                        'regression': 1.0
                    }
                },
                'entropy': {
                    'initial_threshold': 0.5,
                    'decay_rate': 0.05,
                    'min_threshold': 0.1,
                    'task_complexity_factor': 0.2
                },
                'evolution': {
                    'prune_threshold': 0.3,
                    'expand_gamma': 0.1,
                    'mutation_rate': 0.1,
                    'max_layers': 100,
                    'min_layers': 3,
                    'expand_ratio': 1.5 # Added for intelligent expansion
                },
                'architecture': {
                    'depth_decay_lambda': 0.03,
                    'transformer_decay_lambda': 0.01,
                    'discrete_param_range': [1, 10]
                },
                'training': {
                    'info_retention_alpha': 0.7,
                    'structure_variation_beta': 0.3,
                    'convergence_threshold': 1e-4,
                    'max_epochs': 1000
                },
                'monitoring': {
                    'log_interval': 10,
                    'save_interval': 50,
                    'metrics': [
                        'entropy',
                        'mutual_information',
                        'structural_complexity',
                        'layer_importance'
                    ]
                }
            }
            
        return config
        
    def _init_operators(self):
        """Initialize structural operators."""
        # Try to use intelligent operators if available
        try:
            from .core.intelligent_operators import (
                IntelligentExpansionOperator,
                AdaptiveDataFlowOperator,
                BranchSpecializationOperator,
                LayerTypeSelector
            )
            
            # Initialize intelligent operators
            self.layer_selector = LayerTypeSelector(self.info_theory)
            
            self.intelligent_expand_op = IntelligentExpansionOperator(
                layer_selector=self.layer_selector,
                expansion_ratio=self.config['evolution']['expand_ratio'],
                device=self.device
            )
            
            self.data_flow_op = AdaptiveDataFlowOperator(
                min_spatial_size=7,
                complexity_threshold=0.3,
                device=self.device
            )
            
            self.branch_op = BranchSpecializationOperator(
                num_branches=3,
                device=self.device
            )
            
            # Use intelligent operators in compound operator
            self.compound_op = CompoundOperator([
                self.prune_op,
                self.intelligent_expand_op,
                self.data_flow_op,
                self.mutate_op,
                self.branch_op
            ])
            
            self.use_intelligent_operators = True
            
        except ImportError:
            # Fall back to standard operators
            self.prune_op = PruneByEntropy(
                threshold=self.config['evolution']['prune_threshold']
            )
            
            self.expand_op = ExpandWithMI(
                gamma=self.config['evolution']['expand_gamma']
            )
            
            self.mutate_op = MutateDiscrete(
                mutation_rate=self.config['evolution']['mutation_rate']
            )
            
            # Compound operator for complex strategies
            self.compound_op = CompoundOperator([
                self.prune_op,
                self.expand_op,
                self.mutate_op
            ])
            
            self.use_intelligent_operators = False
        
    def wrap_model(self, model: nn.Module) -> nn.Module:
        """
        Wrap a PyTorch model for Neuro Exapt optimization.
        
        Args:
            model: PyTorch model to wrap
            
        Returns:
            Wrapped model with hooks and monitoring
        """
        self.model = model
        self.model.to(self.device)
        
        # Register model with components
        self.struct_optimizer.register_model(model)
        self.info_theory.register_hooks(model)
        
        # Create wrapped model with forward hooks
        self.wrapped_model = NeuroExaptWrapper(
            model,
            self.info_theory,
            self.entropy_ctrl,
            self.info_weight
        )
        
        # Estimate initial complexity
        if self.depth is None:
            complexity = calculate_network_complexity(model)
            self.depth = complexity['depth']
            
        if self.verbose:
            self.logger.info(f"Wrapped model with {self.depth} layers")
            
        return self.wrapped_model
        
    def analyze_model(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_batches: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of the current model.
        
        Args:
            dataloader: Data loader for analysis
            num_batches: Number of batches to analyze
            
        Returns:
            Dictionary of analysis results
        """
        if self.model is None:
            raise ValueError("No model wrapped. Call wrap_model() first.")
            
        # Information-theoretic analysis
        info_analysis = self.info_theory.analyze_network(
            self.model,
            dataloader,
            self.task_type
        )
        
        # Complexity analysis
        complexity_analysis = calculate_network_complexity(
            self.model,
            input_shape=(3, 224, 224)  # Default shape, should be configurable
        )
        
        # Entropy analysis
        entropy_metrics = self._analyze_entropy(dataloader, num_batches)
        
        # Combine results
        analysis = {
            'information': info_analysis,
            'complexity': complexity_analysis,
            'entropy': entropy_metrics,
            'redundancy': calculate_redundancy_score(
                list(info_analysis['layer_importances'].values()),
                info_analysis['output_entropy'],
                self.depth,
                self.config['architecture']['depth_decay_lambda']
            )
        }
        
        return analysis
        
    def _analyze_entropy(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_batches: Optional[int] = None
    ) -> Dict[str, float]:
        """Analyze entropy metrics."""
        entropy_values = []
        layer_entropies = {}
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(dataloader):
                if num_batches and batch_idx >= num_batches:
                    break
                    
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                
                # Measure entropy
                entropy = self.entropy_ctrl.measure(outputs)
                entropy_values.append(entropy)
                
                # Get layer entropies
                for name, value in self.entropy_ctrl.get_layer_entropies().items():
                    if name not in layer_entropies:
                        layer_entropies[name] = []
                    layer_entropies[name].append(value)
                    
        # Average metrics
        avg_entropy = np.mean(entropy_values) if entropy_values else 0.0
        metrics = {
            'network_entropy': avg_entropy,  # 添加这个键
            'average_entropy': avg_entropy,
            'entropy_std': np.std(entropy_values) if entropy_values else 0.0,
            'layer_entropies': {
                name: np.mean(values) 
                for name, values in layer_entropies.items()
            },
            'entropy_trend': self._calculate_entropy_trend(entropy_values),
            'branch_diversity': self._calculate_branch_diversity(layer_entropies)
        }
        
        return metrics
    
    def _calculate_entropy_trend(self, entropy_values: List[float]) -> float:
        """Calculate entropy trend over recent values."""
        if len(entropy_values) < 5:
            return 0.0
        
        # Use last 10 values for trend calculation
        recent_values = entropy_values[-10:]
        if len(recent_values) < 3:
            return 0.0
            
        # Linear regression slope
        x = np.arange(len(recent_values))
        slope = np.polyfit(x, recent_values, 1)[0]
        return float(slope)
    
    def _calculate_branch_diversity(self, layer_entropies: Dict[str, List[float]]) -> float:
        """Calculate diversity between different network branches."""
        if not layer_entropies:
            return 0.0
        
        # Calculate average entropy for each layer
        avg_layer_entropies = {
            name: np.mean(values) 
            for name, values in layer_entropies.items()
        }
        
        if len(avg_layer_entropies) < 2:
            return 0.0
        
        # Calculate standard deviation as diversity measure
        diversity = np.std(list(avg_layer_entropies.values()))
        return float(diversity)
        
    def evolve_structure(
        self,
        performance_metrics: Dict[str, float],
        force_action: Optional[str] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Evolve model structure based on current metrics.
        
        Args:
            performance_metrics: Current performance metrics
            force_action: Force specific action (prune/expand/mutate)
            
        Returns:
            Tuple of (structure_changed, evolution_info)
        """
        if self.model is None:
            raise ValueError("No model wrapped. Call wrap_model() first.")
            
        # Get current metrics
        current_entropy = self.entropy_ctrl.entropy_history[-1] if self.entropy_ctrl.entropy_history else 0.5
        threshold = self.entropy_ctrl.threshold
        
        # Prepare metrics for evolution
        info_metrics = {
            'layer_importances': self.info_theory._activation_cache.get('layer_importances', {}),
            'layer_entropies': self.entropy_ctrl.get_layer_entropies()
        }
        
        entropy_metrics = {
            'current_entropy': current_entropy,
            'threshold': threshold,
            'layer_entropies': info_metrics['layer_entropies']
        }
        
        # Determine action
        if force_action:
            action = force_action
        else:
            if self.entropy_ctrl.should_prune(current_entropy):
                action = 'prune'
            elif self.entropy_ctrl.should_expand(current_entropy):
                action = 'expand'
            elif self.current_epoch % 10 == 0:  # Periodic mutation
                action = 'mutate'
            else:
                action = 'none'
                
        # Apply evolution
        if action != 'none':
            # Use structural evolution
            evolved_model, evolution_step = self.struct_optimizer.evolve_step(
                self.model,
                info_metrics,
                entropy_metrics,
                performance_metrics
            )
            
            # Update model
            self.model = evolved_model
            self.wrapped_model.model = evolved_model
            
            # Re-register hooks
            self.info_theory.clear_hooks()
            self.info_theory.register_hooks(evolved_model)
            
            # Track evolution
            self.evolution_history.append(evolution_step)
            
            if self.verbose:
                self.logger.info(
                    f"Evolved structure: {action} - "
                    f"Parameters: {evolution_step.parameters_before} -> {evolution_step.parameters_after}"
                )
                
            return True, evolution_step.__dict__
            
        return False, {'action': 'none'}
        
    def prune_layers(self, criteria: str = "entropy<0.2") -> List[str]:
        """
        Prune layers based on criteria.
        
        Args:
            criteria: Pruning criteria string
            
        Returns:
            List of pruned layer names
        """
        # Parse criteria
        if "entropy<" in criteria:
            threshold = float(criteria.split("<")[1])
        else:
            threshold = 0.3
            
        # Get layer entropies
        layer_entropies = self.entropy_ctrl.get_layer_entropies()
        
        # Apply pruning
        pruned_model, info = self.prune_op.apply(
            self.model,
            {'layer_entropies': layer_entropies}
        )
        
        if info['pruned_layers']:
            self.model = pruned_model
            self.wrapped_model.model = pruned_model
            
            # Re-register hooks
            self.info_theory.clear_hooks()
            self.info_theory.register_hooks(pruned_model)
            
        return info['pruned_layers']
        
    def expand_layers(
        self,
        method: str = "mutate",
        num_layers: int = 2
    ) -> List[str]:
        """
        Expand network capacity.
        
        Args:
            method: Expansion method
            num_layers: Number of layers to add
            
        Returns:
            List of new layer names
        """
        # Get layer importances
        layer_importances = self.info_theory._activation_cache.get('layer_importances', {})
        
        if method == "mutate":
            # Use mutation operator
            mutated_model, info = self.mutate_op.apply(
                self.model,
                {'layer_importances': layer_importances}
            )
            
            self.model = mutated_model
            self.wrapped_model.model = mutated_model
            
            return info.get('mutations', [])
            
        else:
            # Use expansion operator
            expanded_model, info = self.expand_op.apply(
                self.model,
                {'layer_importances': layer_importances}
            )
            
            if info['expanded_layers']:
                self.model = expanded_model
                self.wrapped_model.model = expanded_model
                
                # Re-register hooks
                self.info_theory.clear_hooks()
                self.info_theory.register_hooks(expanded_model)
                
            return info['expanded_layers']
            
    def update_epoch(self, epoch: int, total_epochs: int):
        """Update current epoch and adapt thresholds."""
        self.current_epoch = epoch
        
        # Update entropy threshold
        self.entropy_ctrl.update_threshold(epoch)
        
        # Update information bottleneck beta if adaptive
        if hasattr(self.info_theory, 'update_beta'):
            self.info_theory.update_beta(epoch, total_epochs)
            
    def monitor(
        self,
        metrics: Optional[List[str]] = None,
        log_dir: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Get monitoring metrics.
        
        Args:
            metrics: List of metrics to monitor
            log_dir: Directory to save logs
            
        Returns:
            Dictionary of metric histories
        """
        if metrics is None:
            metrics = self.config['monitoring']['metrics']
            
        monitored = {}
        
        if 'entropy' in metrics:
            monitored['entropy'] = self.entropy_ctrl.entropy_history
            
        if 'mutual_information' in metrics:
            # Get from info theory component
            monitored['mutual_information'] = []  # Would be populated during training
            
        if 'structural_complexity' in metrics:
            complexity = calculate_network_complexity(self.model)
            monitored['parameters'] = [complexity['total_parameters']]
            
        if 'layer_importance' in metrics:
            importances = self.info_theory._activation_cache.get('layer_importances', {})
            monitored['layer_importances'] = importances
            
        # Log metrics if directory provided
        if log_dir:
            log_metrics(monitored, log_dir, self.current_epoch)
            
        return monitored
        
    def save_state(self, save_path: str):
        """Save complete Neuro Exapt state."""
        state = {
            'model_state': self.model.state_dict() if self.model else None,
            'entropy_controller': self.entropy_ctrl.__dict__,
            'current_epoch': self.current_epoch,
            'evolution_history': self.evolution_history,
            'metrics_history': self.metrics_history,
            'config': self.config
        }
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(state, save_path)
        
        if self.verbose:
            self.logger.info(f"Saved state to {save_path}")
            
    def load_state(self, load_path: str):
        """Load Neuro Exapt state."""
        state = torch.load(load_path, map_location=self.device)
        
        if self.model and state['model_state']:
            self.model.load_state_dict(state['model_state'])
            
        self.entropy_ctrl.__dict__.update(state['entropy_controller'])
        self.current_epoch = state['current_epoch']
        self.evolution_history = state['evolution_history']
        self.metrics_history = state['metrics_history']
        
        if self.verbose:
            self.logger.info(f"Loaded state from {load_path}")
            
    def visualize_evolution(self, save_path: Optional[str] = None):
        """Visualize evolution history."""
        if not self.evolution_history:
            warnings.warn("No evolution history to visualize")
            return
            
        plot_evolution_history(self.evolution_history, save_path)

    def analyze_layer_characteristics(self, model: nn.Module, dataloader: torch.utils.data.DataLoader) -> Dict[str, Any]:
        """
        Analyze layer characteristics for intelligent evolution decisions.
        
        Args:
            model: Model to analyze
            dataloader: Data loader for analysis
            
        Returns:
            Dictionary of layer characteristics
        """
        model.eval()
        layer_characteristics = {}
        
        # Register hooks to capture activations
        activations = {}
        def hook_fn(name):
            def hook(module, input, output):
                activations[name] = output.detach()
            return hook
        
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and len(list(module.parameters())) > 0:
                handle = module.register_forward_hook(hook_fn(name))
                hooks.append(handle)
        
        with torch.no_grad():
            # Run a few batches to collect statistics
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                if batch_idx >= 5:  # Analyze first 5 batches
                    break
                    
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                
                # Analyze each layer's activation
                for name, activation in activations.items():
                    if name not in layer_characteristics:
                        layer_characteristics[name] = {
                            'spatial_complexity': [],
                            'channel_redundancy': [],
                            'activation_sparsity': [],
                            'information_density': []
                        }
                    
                    # Spatial complexity (for conv layers)
                    if activation.dim() == 4:  # Conv2d output
                        # Gradient-based complexity
                        dx = activation[:, :, 1:, :] - activation[:, :, :-1, :]
                        dy = activation[:, :, :, 1:] - activation[:, :, :, :-1]
                        grad_mag = torch.sqrt(dx[:, :, :, :-1]**2 + dy[:, :, :-1, :]**2)
                        complexity = grad_mag.mean().item() / (activation.std().item() + 1e-6)
                        layer_characteristics[name]['spatial_complexity'].append(complexity)
                        
                        # Channel redundancy
                        channels = activation.size(1)
                        if channels > 1:
                            correlations = []
                            for i in range(min(10, channels)):
                                for j in range(i+1, min(10, channels)):
                                    corr = F.cosine_similarity(
                                        activation[:, i].flatten().unsqueeze(0),
                                        activation[:, j].flatten().unsqueeze(0)
                                    ).item()
                                    correlations.append(abs(corr))
                            redundancy = np.mean(correlations) if correlations else 0.0
                            layer_characteristics[name]['channel_redundancy'].append(redundancy)
                    
                    # Activation sparsity
                    sparsity = (activation == 0).float().mean().item()
                    layer_characteristics[name]['activation_sparsity'].append(sparsity)
                    
                    # Information density (entropy-based)
                    entropy = self.info_theory.calculate_entropy(activation)
                    layer_characteristics[name]['information_density'].append(entropy)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Aggregate statistics
        for name in layer_characteristics:
            for metric in layer_characteristics[name]:
                values = layer_characteristics[name][metric]
                if values:
                    layer_characteristics[name][metric] = np.mean(values)
                else:
                    layer_characteristics[name][metric] = 0.0
        
        return layer_characteristics


class NeuroExaptWrapper(nn.Module):
    """
    Wrapper module that adds information-theoretic loss to any model.
    """
    
    def __init__(
        self,
        model: nn.Module,
        info_bottleneck: InformationBottleneck,
        entropy_controller: AdaptiveEntropy,
        info_weight: float = 0.1
    ):
        """
        Initialize wrapper.
        
        Args:
            model: Base model to wrap
            info_bottleneck: Information bottleneck component
            entropy_controller: Entropy controller
            info_weight: Weight for information loss
        """
        super().__init__()
        self.model = model
        self.info_bottleneck = info_bottleneck
        self.entropy_controller = entropy_controller
        self.info_weight = info_weight
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with monitoring."""
        # Standard forward
        output = self.model(x)
        
        # Monitor entropy
        self.entropy_controller.measure(output)
        
        return output
        
    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        base_loss_fn: Callable
    ) -> torch.Tensor:
        """
        Compute combined loss with information-theoretic terms.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            base_loss_fn: Base loss function
            
        Returns:
            Combined loss
        """
        # Base loss
        base_loss = base_loss_fn(predictions, targets)
        
        # Information bottleneck loss
        info_loss = self.info_bottleneck.compute_information_loss(
            predictions,
            targets
        )
        
        # Combined loss
        total_loss = base_loss + self.info_weight * info_loss
        
        return total_loss 