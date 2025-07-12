#!/usr/bin/env python3
"""
NeuroExapt V2 - Deep Classification Example
This example demonstrates the V2 system with a deeper ResNet-like architecture.
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


class ResidualBlock(nn.Module):
    """Basic residual block"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = F.relu(out)
        
        return out


class DeepCNN(nn.Module):
    """Deep CNN with residual connections"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Residual layers
        self.layer1 = self._make_layer(32, 32, 2, stride=1)
        self.layer2 = self._make_layer(32, 64, 2, stride=2)
        self.layer3 = self._make_layer(64, 128, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


def create_dummy_data():
    """Create dummy CIFAR-10 like data for testing"""
    print("Creating dummy data...")
    
    # Create 2000 samples of 32x32 RGB images
    X = torch.randn(2000, 3, 32, 32)
    y = torch.randint(0, 10, (2000,))
    
    # Split into train/test
    train_X, test_X = X[:1600], X[1600:]
    train_y, test_y = y[:1600], y[1600:]
    
    train_dataset = TensorDataset(train_X, train_y)
    test_dataset = TensorDataset(test_X, test_y)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, test_loader


def main():
    print("🧠 NeuroExapt V2 - Deep Classification")
    print("=" * 50)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create base model
    base_model = DeepCNN(num_classes=10)
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    trainer = TrainerV2(model, optimizer, device)
    print("✓ Trainer created")
    
    print("\n🚀 Training for 5 epochs...")
    print("Evolution checks happen automatically every epoch")
    print("Deep networks may trigger more evolution events")
    print("-" * 50)
    
    for epoch in range(5):
        print(f"\nEpoch {epoch + 1}:")
        train_loss, train_acc = trainer.train_epoch(train_loader, epoch)
        val_loss, val_acc = trainer.validate(test_loader)
        print(f"  Train: {train_acc:.1f}% | Val: {val_acc:.1f}% | Loss: {train_loss:.3f}")
    
    print("\n✅ Deep V2 example completed!")


if __name__ == "__main__":
    main()