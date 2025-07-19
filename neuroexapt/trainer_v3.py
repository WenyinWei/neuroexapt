"""
defgroup group_trainer_v3 Trainer V3
ingroup core
Trainer V3 module for NeuroExapt framework.
"""

Trainer for the NeuroExapt V3 framework.

This module provides a `Trainer` class that encapsulates the logic for
training a model using the new information-theoretic and Bayesian-guided
NeuroExapt framework.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any

from .neuroexapt import NeuroExapt
from .core.operators import StructuralOperator, PruneByEntropy, ExpandWithMI
from .math.optimization import AdaptiveLearningRateScheduler

class TrainerV3:
    """
    A trainer to handle the training and evolution loop for a model
    using the NeuroExapt V3 framework.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        operators: List[StructuralOperator],
        device: Optional[torch.device] = None
    ):
        """
        Args:
            model: The base neural network model.
            train_loader: DataLoader for the training set.
            val_loader: DataLoader for the validation set.
            criterion: The loss function.
            optimizer: The optimizer for model parameters.
            operators: A list of operators for evolution.
            device: The device to run on.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize the NeuroExapt instance
        self.neuroexapt = NeuroExapt(
            model=self.model,
            criterion=self.criterion,
            dataloader=self.train_loader, # Use train loader for analysis
            operators=operators,
            device=self.device
        )
        
        # Get the potentially wrapped model from NeuroExapt
        self.model = self.neuroexapt.get_model()

    def evaluate(self) -> Dict[str, float]:
        """Evaluates the model on the validation set."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, targets in self.val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(self.val_loader)
        return {'loss': avg_loss, 'accuracy': accuracy}

    def fit(self, epochs: int, evolution_frequency: int = 1):
        """
        Main training and evolution loop.

        Args:
            epochs: Total number of epochs to train.
            evolution_frequency: How often to run the evolution step.
        """
        print(f"--- Starting TrainerV3 fit for {epochs} epochs ---")
        
        lr_scheduler = AdaptiveLearningRateScheduler(self.optimizer, initial_lr=self.optimizer.param_groups[0]['lr'])

        for epoch in range(epochs):
            self.neuroexapt.current_epoch = epoch + 1
            
            # 1. Train for one epoch
            self.neuroexapt.train_epoch(self.optimizer)
            
            # 2. Evaluate performance
            performance = self.evaluate()
            print(f"Epoch {self.neuroexapt.current_epoch} Validation: Loss={performance['loss']:.4f}, Accuracy={performance['accuracy']:.2f}%")

            # 3. Potentially evolve the architecture
            if self.neuroexapt.current_epoch % evolution_frequency == 0:
                self.neuroexapt.evolve(performance, self.optimizer)

            # 4. Update learning rate
            avg_entropy = self.neuroexapt.evolution_engine.entropy_metrics.get_average_entropy()
            # A proper KL divergence would be computed after an evolution step
            kl_div = 0.0
            lr_scheduler.step(entropy=avg_entropy, kl_divergence=kl_div)
            print(f"Updated LR to: {lr_scheduler.get_lr():.6f}")

        print("--- TrainerV3 fit finished ---")
        return self.neuroexapt.get_model()