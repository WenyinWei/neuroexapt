"""
defgroup group_neuroexapt Neuroexapt
ingroup core
Neuroexapt module for NeuroExapt framework.
"""

NeuroExapt: A Framework for Information-Theoretic and Bayesian-Guided
Dynamic Neural Network Architecture Optimization.
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any, Callable, Tuple

from .core.structural_evolution import StructuralEvolution
from .core.operators import StructuralOperator
from .math.optimization import AdaptiveLearningRateScheduler

# Import visualization tools if available
try:
    from .utils.visualization import ModelVisualizer, Colors
    _VISUALIZATION_AVAILABLE = True
except ImportError:
    _VISUALIZATION_AVAILABLE = False
    # Create a fallback object if visualization is not available
    class _Colors:
        BOLD = ''
        CYAN = ''
        GREEN = ''
        RED = ''
        RESET = ''
    Colors = _Colors()

class NeuroExapt:
    """
    The main NeuroExapt class, which acts as a high-level wrapper for the
    dynamic neural network evolution framework.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        dataloader: DataLoader,
        operators: Optional[List[Any]] = None,
        device: Optional[torch.device] = None,
        lambda_entropy: float = 0.01,
        lambda_bayesian: float = 0.01,
        input_shape: Tuple[int, ...] = (3, 32, 32),
        enable_validation: bool = True,
        evolution_mode: str = "structural",
        use_amp: bool = False,
        evolution_kwargs: Optional[Dict[str, Any]] = None
    ):
        """
        Initializes the NeuroExapt framework.

        Args:
            model: The base neural network model to be evolved.
            criterion: The loss function for the primary task (e.g., CrossEntropyLoss).
            dataloader: DataLoader for the training dataset, used for analysis.
            operators: A list of structural operators available for evolution.
            device: The device to run the model on (e.g., 'cuda' or 'cpu').
            lambda_entropy: Weight for the entropy penalty in the loss function.
            lambda_bayesian: Weight for the Bayesian regularization (KL divergence) in the loss.
            input_shape: Expected input shape for validation.
            enable_validation: Whether to enable dimension validation.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = criterion
        self.dataloader = dataloader
        self.evolution_mode = evolution_mode.lower()
        self.operators = operators  # May be None, will assign defaults below

        # Prepare evolution-specific kwargs
        evolution_kwargs = evolution_kwargs or {}

        self.input_shape = input_shape
        self.enable_validation = enable_validation
        
        self.lambda_entropy = lambda_entropy
        # AMP settings
        self.use_amp = use_amp and torch.cuda.is_available()
        self.scaler = GradScaler(enabled=self.use_amp)
        self.lambda_bayesian = lambda_bayesian

        # Initialize the chosen evolution engine
        if self.evolution_mode == "radical":
            from .core.radical_evolution import RadicalEvolutionEngine
            from .core.radical_operators import get_radical_operator_pool

            if self.operators is None:
                self.operators = get_radical_operator_pool()

            self.evolution_engine = RadicalEvolutionEngine(
                model=self.model,
                operators=self.operators,
                input_shape=self.input_shape,
                enable_validation=self.enable_validation,
                **evolution_kwargs
            )
        else:
            # Default to structural evolution (conservative)
            if self.operators is None:
                # Lazy import to avoid circular deps; fall back to empty list if helper not available
                try:
                    from .core.operators import get_default_operator_pool  # type: ignore
                    self.operators = get_default_operator_pool()
                except (ImportError, AttributeError):
                    self.operators = []

            self.evolution_engine = StructuralEvolution(
                model=self.model,
                operators=self.operators,
                input_shape=self.input_shape,
                enable_validation=self.enable_validation,
                **evolution_kwargs
            )

        # Visualization
        if _VISUALIZATION_AVAILABLE:
            self.visualizer = ModelVisualizer()
        
        self.current_epoch = 0
        self.stats = {
            'evolutions': 0,
            'rollbacks': 0,
            'validation_failures': 0,
            'successful_evolutions': 0
        }

    def _calculate_objective(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor
        ) -> torch.Tensor:
        """
        Calculates the full objective function as defined in the theoretical framework.
        L = L_task + lambda_1 * H(layers) + lambda_2 * D_KL(posterior || prior)
        """
        # 1. Task Loss (e.g., Cross-Entropy)
        task_loss = self.criterion(predictions, targets)

        # 2. Entropy Penalty (Redundancy Penalty)
        # This requires calculating the average entropy of the network.
        # Note: This is an approximation. A more rigorous approach would sum the
        # entropy of each layer's activation distribution.
        avg_entropy = self.evolution_engine.entropy_metrics.get_average_entropy()
        entropy_penalty = self.lambda_entropy * (1.0 - avg_entropy) # Penalize low entropy

        # 3. Bayesian Regularization (KL Divergence)
        # This term is typically calculated within Bayesian layers. If the model
        # uses them, we would sum the KL terms here. For now, this is a placeholder.
        bayesian_loss = torch.tensor(0.0, device=self.device)
        if hasattr(self.model, 'calculate_kl_divergence'):
            # The attribute should be a tensor, not a method to call
            kl_div = getattr(self.model, 'calculate_kl_divergence')
            if isinstance(kl_div, torch.Tensor):
                bayesian_loss = kl_div
        
        bayesian_penalty = self.lambda_bayesian * bayesian_loss

        total_loss = task_loss + entropy_penalty + bayesian_penalty
        return total_loss

    def train_epoch(self, optimizer: torch.optim.Optimizer):
        """
        Runs a single epoch of training, including the calculation of the
        full NeuroExapt objective function.
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        # Register hooks to capture activations for entropy calculation
        self.evolution_engine.entropy_metrics.register_hooks()

        try:
            for data, targets in self.dataloader:
                data, targets = data.to(self.device), targets.to(self.device)

                optimizer.zero_grad(set_to_none=True)

                with autocast(enabled=self.use_amp):
                    predictions = self.model(data)
                    loss = self._calculate_objective(predictions, targets)

                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(predictions.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        except Exception as e:
            print(f"Warning: Training error: {e}")
        finally:
            # Always cleanup hooks after epoch
            self.evolution_engine.entropy_metrics.remove_hooks()
        
        accuracy = 100 * correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(self.dataloader) if len(self.dataloader) > 0 else 0.0
        print(f"Epoch {self.current_epoch} Training: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%")

    def evolve(self, performance_metrics: Dict[str, Any], optimizer: torch.optim.Optimizer):
        """
        Triggers one step of architectural evolution with enhanced error handling.
        """
        try:
            # Dynamically resolve interface to avoid static attribute errors
            step_fn = getattr(self.evolution_engine, "step", None)
            evolve_fn = getattr(self.evolution_engine, "evolve", None)

            if callable(step_fn):
                result = step_fn(
                    epoch=self.current_epoch,
                    performance_metrics=performance_metrics,
                    dataloader=self.dataloader,
                    criterion=self.criterion
                )
            elif callable(evolve_fn):
                result = evolve_fn(
                    epoch=self.current_epoch,
                    dataloader=self.dataloader,
                    criterion=self.criterion,
                    performance_metrics=performance_metrics
                )
            else:
                raise RuntimeError("Evolution engine does not expose compatible step/evolve method.")

            # Normalize result to tuple (model, action_taken)
            if isinstance(result, tuple) and len(result) == 2:
                new_model, action_taken = result  # type: ignore
            else:
                new_model, action_taken = result, None
            
            if action_taken:
                self.model = new_model
                self.stats['evolutions'] += 1
                self.stats['successful_evolutions'] += 1
                
                # CRITICAL: Reset optimizer state after architecture changes
                print("Resetting optimizer state for new architecture...")
                optimizer.state.clear()
                
                # Remove old parameters and add new ones
                optimizer.param_groups.clear()
                optimizer.add_param_group({'params': list(self.model.parameters())})
                
                # Reset cooldown/patience depending on engine implementation
                reset_patience = getattr(self.evolution_engine, "reset_patience", None)
                reset_cooldown = getattr(self.evolution_engine, "reset_cooldown", None)
                if callable(reset_patience):
                    reset_patience()
                elif callable(reset_cooldown):
                    reset_cooldown()
            
            return action_taken
            
        except Exception as e:
            print(f"Evolution error: {e}")
            self.stats['validation_failures'] += 1
            return None

    def fit(self,
            optimizer: torch.optim.Optimizer,
            epochs: int,
            lr_scheduler: Optional[Any] = None,
            evolution_frequency: int = 1,
            test_dataloader: Optional[DataLoader] = None):
        """
        The main training loop for the NeuroExapt framework with enhanced monitoring.

        Args:
            optimizer: The optimizer for training the model parameters.
            epochs: The total number of epochs to train for.
            lr_scheduler: An optional learning rate scheduler.
            evolution_frequency: How many epochs to wait between evolution steps.
            test_dataloader: Optional test dataloader for evaluation.
        """
        print(f"--- Starting NeuroExapt Training for {epochs} epochs ---")
        print(f"Evolution frequency: every {evolution_frequency} epoch(s)")
        print(f"Validation enabled: {self.enable_validation}")

        for epoch in range(epochs):
            self.current_epoch = epoch + 1
            
            # Train for one epoch
            self.train_epoch(optimizer)

            # Potentially evolve the architecture
            if self.current_epoch % evolution_frequency == 0:
                # Get performance metrics
                performance_metrics = {'val_accuracy': 0.0}
                
                if test_dataloader is not None:
                    # Evaluate on test set
                    self.model.eval()
                    correct = 0
                    total = 0
                    with torch.no_grad():
                        for data, targets in test_dataloader:
                            data, targets = data.to(self.device), targets.to(self.device)
                            outputs = self.model(data)
                            _, predicted = torch.max(outputs.data, 1)
                            total += targets.size(0)
                            correct += (predicted == targets).sum().item()
                    
                    performance_metrics['val_accuracy'] = 100 * correct / total if total > 0 else 0.0
                    print(f"Validation accuracy: {performance_metrics['val_accuracy']:.2f}%")

                # Attempt evolution
                action_taken = self.evolve(performance_metrics, optimizer)

            # Update learning rate scheduler
            if lr_scheduler:
                if isinstance(lr_scheduler, AdaptiveLearningRateScheduler):
                    avg_entropy = self.evolution_engine.entropy_metrics.get_average_entropy()
                    lr_scheduler.step(entropy=avg_entropy, kl_divergence=0.0)
                else:
                    lr_scheduler.step()
        
        print("--- NeuroExapt Training Finished ---")
        print(f"Total evolutions: {self.stats['evolutions']}")
        print(f"Successful evolutions: {self.stats['successful_evolutions']}")
        print(f"Validation failures: {self.stats['validation_failures']}")

    def get_model(self) -> nn.Module:
        """Returns the current, potentially evolved model."""
        return self.model

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the evolution process."""
        stats = self.stats.copy()
        get_mem = getattr(self.evolution_engine, "get_memory_usage", None)
        if callable(get_mem):
            mem_stats = get_mem()
            if isinstance(mem_stats, dict):
                stats.update(mem_stats)
        return stats

    def cleanup(self):
        """Clean up all resources."""
        self.evolution_engine.cleanup()

    def __repr__(self) -> str:
        return f"NeuroExapt(model={self.model.__class__.__name__}, evolutions={self.stats['evolutions']})"