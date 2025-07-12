"""
NeuroExapt V3 Trainer - Simplified Interface

This module provides a simplified interface for using NeuroExapt V3 with
minimal configuration required from users.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings

from .neuroexapt_v3 import NeuroExaptV3


class TrainerV3:
    """
    Simplified trainer interface for NeuroExapt V3.
    
    Provides easy-to-use methods for training neural networks with
    adaptive architecture evolution based on information theory.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        efficiency_threshold: float = 0.1,
        verbose: bool = True,
        log_directory: str = "./neuroexapt_logs"
    ):
        """
        Initialize NeuroExapt V3 trainer.
        
        Args:
            model: PyTorch model to train and optimize
            device: Device for training (auto-detected if None)
            efficiency_threshold: Minimum efficiency gain for evolution
            verbose: Whether to show evolution visualizations
            log_directory: Directory for saving logs
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize NeuroExapt V3 system
        self.neuroexapt = NeuroExaptV3(
            model=model,
            device=self.device,
            efficiency_threshold=efficiency_threshold,
            enable_visualization=verbose,
            log_directory=log_directory
        )
        
        # Training configuration
        self.verbose = verbose
        self.model = self.neuroexapt.model
        
        # Training state
        self.history = None
        self.best_accuracy = 0.0
        self.best_model_state = None
        
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        learning_rate: float = 0.001,
        optimizer_type: str = 'adam',
        criterion_type: str = 'crossentropy',
        scheduler_type: Optional[str] = None,
        **kwargs
    ) -> Dict[str, List[float]]:
        """
        Train the model with adaptive evolution.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            optimizer_type: Type of optimizer ('adam', 'sgd', 'rmsprop')
            criterion_type: Type of loss function ('crossentropy', 'mse')
            scheduler_type: Optional learning rate scheduler
            **kwargs: Additional arguments
            
        Returns:
            Training history dictionary
        """
        # Setup optimizer
        optimizer = self._create_optimizer(optimizer_type, learning_rate)
        
        # Setup loss function
        criterion = self._create_criterion(criterion_type)
        
        if self.verbose:
            print(f"🧠 Starting NeuroExapt V3 training:")
            print(f"  - Model: {self.model.__class__.__name__}")
            print(f"  - Epochs: {epochs}")
            print(f"  - Learning rate: {learning_rate}")
            print(f"  - Optimizer: {optimizer_type}")
            print(f"  - Device: {self.device}")
        
        # Train with adaptive evolution
        self.history = self.neuroexapt.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            optimizer=optimizer,
            criterion=criterion,
            **kwargs
        )
        
        # Track best model
        best_val_acc = max(self.history['val_accuracy'])
        if best_val_acc > self.best_accuracy:
            self.best_accuracy = best_val_acc
            self.best_model_state = self.model.state_dict().copy()
        
        # Print final summary
        if self.verbose:
            final_train_acc = self.history['train_accuracy'][-1]
            final_val_acc = self.history['val_accuracy'][-1]
            total_evolutions = sum(self.history['evolutions'])
            
            print(f"\n✅ Training completed!")
            print(f"  - Final training accuracy: {final_train_acc:.2f}%")
            print(f"  - Final validation accuracy: {final_val_acc:.2f}%")
            print(f"  - Best validation accuracy: {self.best_accuracy:.2f}%")
            print(f"  - Total architecture evolutions: {total_evolutions}")
        
        return self.history
    
    def _create_optimizer(self, optimizer_type: str, learning_rate: float) -> optim.Optimizer:
        """Create optimizer based on type."""
        optimizer_type = optimizer_type.lower()
        
        if optimizer_type == 'adam':
            return optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_type == 'sgd':
            return optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        elif optimizer_type == 'rmsprop':
            return optim.RMSprop(self.model.parameters(), lr=learning_rate)
        elif optimizer_type == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=learning_rate)
        else:
            warnings.warn(f"Unknown optimizer type: {optimizer_type}, using Adam")
            return optim.Adam(self.model.parameters(), lr=learning_rate)
    
    def _create_criterion(self, criterion_type: str) -> nn.Module:
        """Create loss function based on type."""
        criterion_type = criterion_type.lower()
        
        if criterion_type == 'crossentropy':
            return nn.CrossEntropyLoss()
        elif criterion_type == 'mse':
            return nn.MSELoss()
        elif criterion_type == 'bce':
            return nn.BCELoss()
        elif criterion_type == 'bcewithlogits':
            return nn.BCEWithLogitsLoss()
        else:
            warnings.warn(f"Unknown criterion type: {criterion_type}, using CrossEntropyLoss")
            return nn.CrossEntropyLoss()
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Evaluation metrics
        """
        self.model.eval()
        correct = 0
        total = 0
        test_loss = 0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        test_accuracy = 100. * correct / total
        avg_test_loss = test_loss / len(test_loader)
        
        if self.verbose:
            print(f"📊 Test Results:")
            print(f"  - Test accuracy: {test_accuracy:.2f}%")
            print(f"  - Test loss: {avg_test_loss:.4f}")
        
        return {
            'test_accuracy': test_accuracy,
            'test_loss': avg_test_loss
        }
    
    def get_model(self) -> nn.Module:
        """Get the current optimized model."""
        return self.neuroexapt.get_model()
    
    def get_best_model(self) -> nn.Module:
        """Get the best model (highest validation accuracy)."""
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        return self.model
    
    def save_model(self, filepath: str, save_best: bool = True):
        """
        Save the model and training state.
        
        Args:
            filepath: Path to save the model
            save_best: Whether to save the best model or current model
        """
        if save_best and self.best_model_state is not None:
            # Save best model
            torch.save({
                'model_state_dict': self.best_model_state,
                'best_accuracy': self.best_accuracy,
                'history': self.history,
                'evolution_summary': self.neuroexapt.get_evolution_summary()
            }, filepath)
        else:
            # Save current model
            self.neuroexapt.save_model(filepath)
        
        if self.verbose:
            print(f"💾 Model saved to: {filepath}")
    
    def load_model(self, filepath: str):
        """Load a previously saved model."""
        self.neuroexapt.load_model(filepath)
        self.model = self.neuroexapt.get_model()
        if self.verbose:
            print(f"📂 Model loaded from: {filepath}")
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get comprehensive evolution summary."""
        return self.neuroexapt.get_evolution_summary()
    
    def analyze_architecture(self, data_loader: DataLoader) -> Dict[str, Any]:
        """
        Analyze current architecture with information theory metrics.
        
        Args:
            data_loader: Data loader for analysis
            
        Returns:
            Architecture analysis results
        """
        analysis = self.neuroexapt.analyze_architecture(data_loader)
        
        if self.verbose:
            print("🔍 Architecture Analysis:")
            print(f"  - Total redundancy: {analysis['total_redundancy']:.3f}")
            print(f"  - Computational efficiency: {analysis['computational_efficiency']:.3f}")
            print(f"  - Total parameters: {analysis['total_parameters']:,}")
            print(f"  - Trainable parameters: {analysis['trainable_parameters']:,}")
            print(f"  - Conv layers: {analysis['conv_layers']}")
            print(f"  - Linear layers: {analysis['linear_layers']}")
        
        return analysis
    
    def __repr__(self) -> str:
        evolutions = len(self.neuroexapt.evolution_history)
        return f"TrainerV3(model={self.model.__class__.__name__}, evolutions={evolutions})"


# Convenience function for quick training
def train_with_neuroexapt(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    learning_rate: float = 0.001,
    device: Optional[torch.device] = None,
    **kwargs
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Convenience function to train a model with NeuroExapt V3.
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device for training
        **kwargs: Additional arguments for trainer
        
    Returns:
        Tuple of (optimized_model, training_history)
    """
    trainer = TrainerV3(model, device=device, **kwargs)
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        learning_rate=learning_rate
    )
    
    return trainer.get_best_model(), history