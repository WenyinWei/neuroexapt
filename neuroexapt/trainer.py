"""
Trainer module for Neuro Exapt.

This module provides the training loop with integrated information-theoretic
optimization and dynamic architecture evolution.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
from tqdm import tqdm
import time
import warnings
from collections import defaultdict

from .neuroexapt import NeuroExapt
from .utils.logging import setup_logger, log_metrics
from .utils.visualization import plot_evolution_history, plot_entropy_history
from .core.dynarch import DynamicArchitecture


class Trainer:
    """
    Trainer class for Neuro Exapt framework.
    
    Handles training loop with integrated information-theoretic optimization
    and dynamic architecture evolution.
    """
    
    def __init__(
        self,
        model: nn.Module,
        neuro_exapt: Optional[NeuroExapt] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        info_weight: float = 0.1,
        entropy_threshold: float = 0.3,
        evolution_frequency: int = 10,
        device: Optional[torch.device] = None,
        verbose: bool = True
    ):
        """
        Initialize Trainer.
        
        Args:
            model: Neural network model to train
            neuro_exapt: NeuroExapt instance (will create default if None)
            optimizer: Optimizer for training
            scheduler: Learning rate scheduler
            info_weight: Weight for information-theoretic loss
            entropy_threshold: Threshold for entropy-based decisions
            evolution_frequency: Frequency of structure evolution (epochs)
            device: Torch device for training
            verbose: Whether to print training progress
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        
        # Initialize NeuroExapt if not provided
        if neuro_exapt is None:
            self.neuro_exapt = NeuroExapt(
                info_weight=info_weight,
                entropy_weight=entropy_threshold,
                device=self.device,
                verbose=verbose
            )
        else:
            self.neuro_exapt = neuro_exapt
            
        # Wrap model
        self.wrapped_model = self.neuro_exapt.wrap_model(model)
        self.wrapped_model.to(self.device)
        
        # Setup optimizer
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.wrapped_model.parameters(),
                lr=1e-3,
                weight_decay=1e-4
            )
        else:
            self.optimizer = optimizer
            
        self.scheduler = scheduler
        self.evolution_frequency = evolution_frequency
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.best_accuracy = 0.0
        self.current_train_loader = None  # Store train_loader for evolution
        
        # Metrics tracking
        self.training_history = defaultdict(list)
        self.evolution_events = []
        
        # Setup logging
        if self.verbose:
            self.logger = setup_logger("Trainer")
            
        self.dynarch = DynamicArchitecture(
            self.wrapped_model, 
            self.neuro_exapt.struct_optimizer, 
            self.neuro_exapt.info_theory,
            device=self.device  # Pass device to DynamicArchitecture
        )
        
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        loss_fn: Optional[Callable] = None,
        eval_metric: str = "accuracy",
        early_stopping_patience: int = 20,
        save_best: bool = True,
        save_path: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Train the model with dynamic architecture optimization.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            loss_fn: Loss function (will use CrossEntropyLoss for classification)
            eval_metric: Evaluation metric for model selection
            early_stopping_patience: Patience for early stopping
            save_best: Whether to save best model
            save_path: Path to save best model
            
        Returns:
            Training history dictionary
        """
        # Store train_loader for evolution
        self.current_train_loader = train_loader
        
        # Setup loss function
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()
            
        # Early stopping setup
        patience_counter = 0
        best_val_metric = 0.0 if eval_metric == "accuracy" else float('inf')
        
        if self.verbose:
            self.logger.info(f"Starting training for {epochs} epochs")
            self.logger.info(f"Device: {self.device}")
            self.logger.info(f"Model parameters: {sum(p.numel() for p in self.wrapped_model.parameters())}")
            
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Update epoch in NeuroExapt
            self.neuro_exapt.update_epoch(epoch, epochs)
            
            # Training phase
            train_metrics = self._train_epoch(train_loader, loss_fn)
            
            # Validation phase
            if val_loader is not None:
                val_metrics = self._validate_epoch(val_loader, loss_fn)
            else:
                val_metrics = {}
                
            # Log metrics
            self._log_epoch_metrics(epoch, train_metrics, val_metrics)
            
            # Structure evolution
            if epoch % self.evolution_frequency == 0 and epoch > 0:
                self._evolve_structure(train_metrics, val_metrics)
                
            # Learning rate scheduling
            if self.scheduler is not None:
                if hasattr(self.scheduler, 'step'):
                    if 'ReduceLROnPlateau' in str(type(self.scheduler)):
                        val_loss = val_metrics.get('loss', train_metrics['loss'])
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
                        
            # Early stopping check
            current_val_metric = val_metrics.get(eval_metric, train_metrics.get(eval_metric, 0.0))
            
            if eval_metric == "accuracy":
                is_better = current_val_metric > best_val_metric
            else:
                is_better = current_val_metric < best_val_metric
                
            if is_better:
                best_val_metric = current_val_metric
                patience_counter = 0
                
                if save_best and save_path:
                    self._save_checkpoint(save_path, epoch, train_metrics, val_metrics)
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                if self.verbose:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                break
                
        if self.verbose:
            self.logger.info("Training completed")
            
            # Print final DynArch statistics
            stats = self.dynarch.get_stats()
            self.logger.info(f"🤖 DynArch Statistics:")
            self.logger.info(f"   Total evolution steps: {stats['evolution_steps']}")
            self.logger.info(f"   Final epsilon: {stats['current_epsilon']:.3f}")
            self.logger.info(f"   Pareto front size: {stats['pareto_front_size']}")
            self.logger.info(f"   Action distribution: {stats['action_distribution']}")
            
        return dict(self.training_history)
        
    def _train_epoch(self, train_loader: DataLoader, loss_fn: Callable) -> Dict[str, float]:
        """Train for one epoch."""
        self.wrapped_model.train()
        
        total_loss = 0.0
        total_info_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, disable=not self.verbose, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.wrapped_model(data)
            
            # Compute loss
            if hasattr(self.wrapped_model, 'compute_loss'):
                loss = self.wrapped_model.compute_loss(output, target, loss_fn)
            else:
                loss = loss_fn(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            
            # Accuracy calculation (assuming classification)
            if len(target.shape) == 1:  # Classification
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                
            # Update progress bar
            if batch_idx % 50 == 0:
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.0 * correct / total:.2f}%' if total > 0 else 'N/A'
                })
                
        # Calculate epoch metrics
        epoch_loss = total_loss / len(train_loader)
        epoch_accuracy = 100.0 * correct / total if total > 0 else 0.0
        
        metrics = {
            'loss': epoch_loss,
            'accuracy': epoch_accuracy
        }
        
        # Add information-theoretic metrics
        info_metrics = self._calculate_info_metrics()
        metrics.update(info_metrics)
        
        return metrics
        
    def _validate_epoch(self, val_loader: DataLoader, loss_fn: Callable) -> Dict[str, float]:
        """Validate for one epoch."""
        self.wrapped_model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.wrapped_model(data)
                if hasattr(self.wrapped_model, 'compute_loss'):
                    loss = self.wrapped_model.compute_loss(output, target, loss_fn)
                else:
                    loss = loss_fn(output, target)
                
                total_loss += loss.item()
                
                # Accuracy calculation
                if len(target.shape) == 1:
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                    total += target.size(0)
                    
        epoch_loss = total_loss / len(val_loader)
        epoch_accuracy = 100.0 * correct / total if total > 0 else 0.0
        
        return {
            'val_loss': epoch_loss,
            'val_accuracy': epoch_accuracy
        }
        
    def _calculate_info_metrics(self) -> Dict[str, float]:
        """Calculate information-theoretic metrics."""
        metrics = {}
        
        try:
            # Entropy metrics
            entropy_history = self.neuro_exapt.entropy_ctrl.entropy_history
            if entropy_history:
                metrics['entropy'] = entropy_history[-1]
                metrics['entropy_threshold'] = self.neuro_exapt.entropy_ctrl.threshold
                
            # Network complexity
            complexity = self.neuro_exapt.info_theory._activation_cache.get('network_complexity', 0)
            metrics['complexity'] = complexity
            
            # Layer importance variance (measure of information distribution)
            layer_importances = self.neuro_exapt.info_theory._activation_cache.get('layer_importances', {})
            if layer_importances:
                importance_values = list(layer_importances.values())
                metrics['importance_variance'] = np.var(importance_values)
                metrics['importance_mean'] = np.mean(importance_values)
                
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"Could not calculate info metrics: {e}")
                
        return metrics
        
    def _evolve_structure(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ):
        """Perform structure evolution based on current metrics."""
        try:
            # Combine metrics for evolution decision
            performance_metrics = {**train_metrics, **val_metrics}
            
            state = self.dynarch.get_state(performance_metrics)
            action = self.dynarch.select_action(state)
            
            # Create a mini-loader for quick evaluation (use first few batches)
            mini_loader_data = []
            if self.current_train_loader:
                train_iter = iter(self.current_train_loader)
                for _ in range(3):  # Use first 3 batches
                    try:
                        mini_loader_data.append(next(train_iter))
                    except StopIteration:
                        break
            
            success, evolution_info = self.dynarch.apply_tentative(action, state, mini_loader_data)
            
            if success:
                print("🎉 Evolution successful!")
                
                # Update optimizer for new parameters
                self.optimizer = torch.optim.AdamW(
                    self.wrapped_model.parameters(),
                    lr=self.optimizer.param_groups[0]['lr'],
                    weight_decay=self.optimizer.param_groups[0]['weight_decay']
                )
                
                # Log evolution event
                self.evolution_events.append({
                    'epoch': self.current_epoch,
                    'action': evolution_info.get('action_info', {}).get('action_type', 'unknown'),
                    'info': evolution_info
                })
                
                if self.verbose:
                    action_info = evolution_info.get('action_info', {})
                    reward = evolution_info.get('reward', 0)
                    improvements = evolution_info.get('metrics_improvement', {})
                    
                    # Detailed evolution report
                    self.logger.info(f"🔄 ARCHITECTURE EVOLUTION at epoch {self.current_epoch}:")
                    self.logger.info(f"   Action: {action_info.get('action_type', 'unknown')}")
                    self.logger.info(f"   Reward: {reward:.4f}")
                    
                    for metric, change in improvements.items():
                        if change != 0:
                            sign = "+" if change > 0 else ""
                            self.logger.info(f"   {metric.title()} change: {sign}{change:.4f}")
                    
                    self.logger.info("")
                    
                    # Visualize
                    from neuroexapt.utils.visualization import ascii_model_graph
                    ascii_model_graph(self.wrapped_model)
            else:
                reason = evolution_info.get('reason', 'unknown')
                if self.verbose:
                    self.logger.info(f"❌ Evolution failed: {reason}")
                    
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"Structure evolution failed: {e}")
                
    def _log_epoch_metrics(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ):
        """Log metrics for the current epoch."""
        # Store in history
        for key, value in train_metrics.items():
            self.training_history[key].append(value)
            
        for key, value in val_metrics.items():
            self.training_history[key].append(value)
            
        # Console logging
        if self.verbose and epoch % 10 == 0:
            log_str = f"Epoch {epoch:3d}: "
            log_str += f"Loss: {train_metrics['loss']:.4f} "
            
            if 'accuracy' in train_metrics:
                log_str += f"Acc: {train_metrics['accuracy']:.2f}% "
                
            if val_metrics:
                log_str += f"Val Loss: {val_metrics.get('val_loss', 0):.4f} "
                if 'val_accuracy' in val_metrics:
                    log_str += f"Val Acc: {val_metrics['val_accuracy']:.2f}% "
                    
            if 'entropy' in train_metrics:
                log_str += f"Entropy: {train_metrics['entropy']:.3f} "
                
            self.logger.info(log_str)
            
    def _save_checkpoint(
        self,
        save_path: str,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.wrapped_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'evolution_events': self.evolution_events,
            'neuro_exapt_config': self.neuro_exapt.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        torch.save(checkpoint, save_path)
        
        if self.verbose:
            self.logger.info(f"Saved checkpoint to {save_path}")
            
    def load_checkpoint(self, load_path: str) -> Dict[str, Any]:
        """Load model checkpoint."""
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.wrapped_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        self.current_epoch = checkpoint['epoch']
        self.evolution_events = checkpoint.get('evolution_events', [])
        
        if self.verbose:
            self.logger.info(f"Loaded checkpoint from {load_path}")
            
        return checkpoint
        
    def analyze_model(self, dataloader: DataLoader) -> Dict[str, Any]:
        """Perform comprehensive model analysis."""
        return self.neuro_exapt.analyze_model(dataloader)
        
    def visualize_training(self, save_path: Optional[str] = None):
        """Visualize training progress."""
        if not self.training_history:
            warnings.warn("No training history to visualize")
            return
            
        # TODO: Implement training progress visualization
        if self.verbose:
            self.logger.info("Training visualization not yet implemented")
        
    def visualize_evolution(self, save_path: Optional[str] = None):
        """Visualize structure evolution history."""
        if not self.evolution_events:
            warnings.warn("No evolution events to visualize")
            return
            
        # Create evolution history from events
        evolution_history = []
        for event in self.evolution_events:
            evolution_history.append(event['info'])
            
        plot_evolution_history(evolution_history, save_path)
        
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary."""
        total_params = sum(p.numel() for p in self.wrapped_model.parameters())
        trainable_params = sum(p.numel() for p in self.wrapped_model.parameters() if p.requires_grad)
        
        summary = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'current_epoch': self.current_epoch,
            'evolution_events': len(self.evolution_events),
            'device': str(self.device)
        }
        
        # Add latest metrics
        if self.training_history:
            for key, values in self.training_history.items():
                if values:
                    summary[f'latest_{key}'] = values[-1]
                    
        return summary 