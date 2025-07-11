"""
Basic Classification Example with Neuro Exapt.

This example demonstrates how to use Neuro Exapt for dynamic architecture
optimization on a simple classification task using CIFAR-10.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import neuroexapt


class SimpleCNN(nn.Module):
    """Simple CNN for CIFAR-10 classification."""
    
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling layers
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.25)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten for fully connected layers
        x = x.view(-1, 128 * 4 * 4)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x


def get_cifar10_dataloaders(batch_size=128, num_workers=2):
    """Get CIFAR-10 data loaders with advanced downloading capabilities including 迅雷 integration."""
    
    # Import the advanced dataset loader
    from neuroexapt.utils.dataset_loader import AdvancedDatasetLoader
    
    # Initialize the advanced loader with P2P acceleration, caching, and 迅雷 integration
    loader = AdvancedDatasetLoader(
        cache_dir="./data_cache",      # Cache directory for downloaded files
        download_dir="./data",         # Directory for extracted datasets
        use_p2p=True,                  # Enable P2P acceleration
        use_xunlei=True,               # Enable 迅雷 integration for Chinese users
        max_retries=3                  # Number of retry attempts
    )
    
    # Get data loaders with automatic downloading and caching
    return loader.get_cifar10_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        download=True,                 # Automatically download if not present
        force_download=False           # Don't force re-download if cached
    )


def main():
    """Main training function."""
    
    print("=" * 60)
    print("Neuro Exapt - Basic Classification Example")
    print("=" * 60)
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Hyperparameters
    batch_size = 128
    learning_rate = 0.001
    epochs = 50
    info_weight = 0.1  # Weight for information-theoretic loss
    entropy_threshold = 0.3  # Threshold for structural decisions
    
    # Get data loaders
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_cifar10_dataloaders(batch_size)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("\nInitializing model...")
    model = SimpleCNN(num_classes=10)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Initialize Neuro Exapt
    print("\nInitializing Neuro Exapt...")
    neuro_exapt = neuroexapt.NeuroExapt(
        task_type="classification",
        entropy_weight=entropy_threshold,
        info_weight=info_weight,
        device=device,
        verbose=True
    )
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = neuroexapt.Trainer(
        model=model,
        neuro_exapt=neuro_exapt,
        optimizer=optimizer,
        scheduler=scheduler,
        evolution_frequency=10,  # Evolve structure every 10 epochs
        device=device,
        verbose=True
    )
    
    # Analyze initial model
    print("\nAnalyzing initial model...")
    initial_analysis = trainer.analyze_model(train_loader)
    print(f"Initial complexity: {initial_analysis['complexity']}")
    print(f"Initial entropy: {initial_analysis['entropy'].get('network_entropy', 'N/A')}")
    
    # Training
    print(f"\nStarting training for {epochs} epochs...")
    print("-" * 60)
    
    training_history = trainer.fit(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=epochs,
        loss_fn=nn.CrossEntropyLoss(),
        eval_metric="val_accuracy",
        early_stopping_patience=15,
        save_best=True,
        save_path="./checkpoints/best_model.pth"
    )
    
    # Final analysis
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    
    final_analysis = trainer.analyze_model(test_loader)
    model_summary = trainer.get_model_summary()
    
    print("\nFinal Results:")
    print(f"Final test accuracy: {training_history.get('val_accuracy', [0])[-1]:.2f}%")
    print(f"Total parameters: {model_summary['total_parameters']}")
    print(f"Evolution events: {model_summary['evolution_events']}")
    print(f"Final entropy: {final_analysis['entropy'].get('network_entropy', 'N/A')}")
    
    # Visualizations
    print("\nGenerating visualizations...")
    os.makedirs("./results", exist_ok=True)
    
    try:
        trainer.visualize_training("./results/training_progress.png")
        trainer.visualize_evolution("./results/evolution_history.png")
        neuro_exapt.visualize_evolution("./results/neuro_exapt_evolution.png")
        print("Visualizations saved to ./results/")
    except Exception as e:
        print(f"Could not generate visualizations: {e}")
    
    # Print evolution summary
    if trainer.evolution_events:
        print("\nEvolution Events:")
        for i, event in enumerate(trainer.evolution_events):
            print(f"  {i+1}. Epoch {event['epoch']}: {event['action']}")
    else:
        print("\nNo structure evolution occurred during training.")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main() 