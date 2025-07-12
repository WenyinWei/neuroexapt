#!/usr/bin/env python3
"""
NeuroExapt V3 - Deep Classification Example
This example demonstrates the V3 system with a deeper ResNet-like architecture.
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
    print("🧠 NeuroExapt V3 - Deep Classification")
    print("=" * 50)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create base model
    model = DeepCNN(num_classes=10)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Deep model: {param_count:,} parameters")
    print("✓ Deep ResNet-like model created")
    
    # Get data
    train_loader, test_loader = create_dummy_data()
    print("✓ Data loaded (dummy CIFAR-10)")
    
    # V3 training with intelligent evolution
    print("\n🚀 Training with NeuroExapt V3 intelligent evolution")
    print("Evolution checks happen every epoch, but only change when beneficial")
    print("Deep networks may trigger more sophisticated evolution events")
    print("=" * 60)
    
    # Use convenience function for deep model training
    optimized_model, history = train_with_neuroexapt(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=8,  # More epochs for deep model
        learning_rate=0.001,
        efficiency_threshold=0.08,  # Slightly higher for deep models
        verbose=True
    )
    
    print(f"\n📊 V3 Deep Training Results:")
    print(f"  Final train accuracy: {history['train_accuracy'][-1]:.1f}%")
    print(f"  Final val accuracy: {history['val_accuracy'][-1]:.1f}%")
    print(f"  Total architecture evolutions: {sum(history['evolutions'])}")
    print(f"  Evolution frequency: {sum(history['evolutions'])/len(history['evolutions'])*100:.1f}%")
    
    # Detailed analysis for deep model
    print("\n🔍 Deep Model Analysis:")
    trainer = TrainerV3(model=optimized_model, verbose=False)
    analysis = trainer.analyze_architecture(test_loader)
    print(f"  Computational efficiency: {analysis['computational_efficiency']:.3f}")
    print(f"  Total redundancy: {analysis['total_redundancy']:.3f}")
    print(f"  Conv layers: {analysis['conv_layers']}")
    print(f"  Linear layers: {analysis['linear_layers']}")
    print(f"  Final parameters: {analysis['total_parameters']:,}")
    
    # Evolution summary
    evolution_summary = trainer.get_evolution_summary()
    print(f"\n🧠 Evolution Intelligence Summary:")
    print(f"  Success rate: {evolution_summary['evolution_stats']['success_rate']:.1%}")
    print(f"  No-change rate: {evolution_summary['evolution_stats']['no_change_rate']:.1%}")
    print(f"  Performance trend: {evolution_summary['performance_trend']}")
    
    print("\n✅ Deep V3 example completed!")


if __name__ == "__main__":
    main()