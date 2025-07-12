#!/usr/bin/env python3
"""
NeuroExapt V2 - Basic Classification Example
This example demonstrates the V2 system with automatic evolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

# Add neuroexapt to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuroexapt.core.dynamic_architecture_v2 import DynamicArchitectureV2
from neuroexapt.trainer import TrainerV2


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
    print("🧠 NeuroExapt V2 - Basic Classification")
    print("=" * 50)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create base model
    base_model = SimpleCNN()
    param_count = sum(p.numel() for p in base_model.parameters())
    print(f"Base model: {param_count:,} parameters")
    
    # Wrap with V2 architecture
    model = DynamicArchitectureV2(base_model, device)
    model.to(device)
    print("✓ V2 architecture wrapper created")
    
    # Get data
    train_loader, test_loader = create_dummy_data()
    print("✓ Data loaded (dummy CIFAR-10)")
    
    # Create trainer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    trainer = TrainerV2(model, optimizer, device)
    print("✓ Trainer created")
    
    print("\n🚀 Training for 3 epochs...")
    print("Evolution checks happen automatically every epoch")
    print("-" * 50)
    
    for epoch in range(3):
        print(f"\nEpoch {epoch + 1}:")
        train_loss, train_acc = trainer.train_epoch(train_loader, epoch)
        val_loss, val_acc = trainer.validate(test_loader)
        print(f"  Train: {train_acc:.1f}% | Val: {val_acc:.1f}% | Loss: {train_loss:.3f}")
    
    print("\n✅ V2 example completed!")


if __name__ == "__main__":
    main()