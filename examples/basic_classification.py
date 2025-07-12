#!/usr/bin/env python3
"""
NeuroExapt V3 - Basic Classification Example
This example demonstrates the V3 system with intelligent evolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

# Add neuroexapt to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuroexapt.trainer_v3 import TrainerV3, train_with_neuroexapt


class SimpleCNN(nn.Module):
    """Simple CNN for CIFAR-10 classification"""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def create_dummy_data():
    """Create dummy CIFAR-10 like data for testing"""
    print("Creating dummy data...")
    
    # Create 1000 samples of 32x32 RGB images
    X = torch.randn(1000, 3, 32, 32)
    y = torch.randint(0, 10, (1000,))
    
    # Split into train/test
    train_X, test_X = X[:800], X[800:]
    train_y, test_y = y[:800], y[800:]
    
    train_dataset = TensorDataset(train_X, train_y)
    test_dataset = TensorDataset(test_X, test_y)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, test_loader


def main():
    print("🧠 NeuroExapt V3 - Basic Classification")
    print("=" * 50)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create base model
    model = SimpleCNN()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Base model: {param_count:,} parameters")
    print("✓ Model created")
    
    # Get data
    train_loader, test_loader = create_dummy_data()
    print("✓ Data loaded (dummy CIFAR-10)")
    
    # V3 Method 1: Use convenience function (recommended)
    print("\n🚀 Method 1: One-line training with NeuroExapt V3")
    print("=" * 50)
    
    optimized_model, history = train_with_neuroexapt(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=5,
        learning_rate=0.001,
        efficiency_threshold=0.05,
        verbose=True
    )
    
    print(f"\n📊 V3 Training Results:")
    print(f"  Final train accuracy: {history['train_accuracy'][-1]:.1f}%")
    print(f"  Final val accuracy: {history['val_accuracy'][-1]:.1f}%")
    print(f"  Total evolutions: {sum(history['evolutions'])}")
    
    # V3 Method 2: Detailed control
    print("\n🚀 Method 2: Detailed control with TrainerV3")
    print("=" * 50)
    
    model2 = SimpleCNN()
    trainer = TrainerV3(
        model=model2,
        device=device,
        efficiency_threshold=0.1,
        verbose=True
    )
    
    # Train with custom parameters
    history2 = trainer.fit(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=3,
        learning_rate=0.001,
        optimizer_type='adam'
    )
    
    print(f"\n📊 Detailed Training Results:")
    print(f"  Final train accuracy: {history2['train_accuracy'][-1]:.1f}%")
    print(f"  Final val accuracy: {history2['val_accuracy'][-1]:.1f}%")
    print(f"  Total evolutions: {sum(history2['evolutions'])}")
    
    # Architecture analysis
    analysis = trainer.analyze_architecture(test_loader)
    print(f"\n🔍 Architecture Analysis:")
    print(f"  Total redundancy: {analysis['total_redundancy']:.3f}")
    print(f"  Computational efficiency: {analysis['computational_efficiency']:.3f}")
    print(f"  Total parameters: {analysis['total_parameters']:,}")
    
    print("\n✅ V3 example completed!")


if __name__ == "__main__":
    main()