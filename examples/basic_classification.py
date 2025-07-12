#!/usr/bin/env python3
"""
NeuroExapt - Basic Classification Example (50 epochs)

This example demonstrates NeuroExapt with real CIFAR-10 dataset for 80%+ accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import sys
import os

# Add neuroexapt to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuroexapt.trainer import Trainer, train_with_neuroexapt


class ImprovedCNN(nn.Module):
    """Well-designed CNN with proper regularization to avoid overfitting"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Feature extraction with batch norm and dropout
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            
            # Second block
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3),
            
            # Third block
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Classifier with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 16, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def create_cifar10_dataloaders():
    """Create CIFAR-10 dataloaders using the real dataset"""
    print("Loading CIFAR-10 dataset...")
    
    # Define transforms for CIFAR-10
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Load CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=True,
        transform=transform_train
    )
    
    val_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=True,
        transform=transform_test
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)
    
    print(f"Dataset: {len(train_dataset)} train, {len(val_dataset)} val samples")
    
    return train_loader, val_loader


def main():
    print("🚀 CIFAR-10 Classification with NeuroExapt (50 Epochs)")
    print("=" * 55)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create improved model
    model = ImprovedCNN(num_classes=10)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {param_count:,} parameters")
    
    # Get CIFAR-10 dataset
    train_loader, val_loader = create_cifar10_dataloaders()
    
    # Train with V3 for 50 epochs
    optimized_model, history = train_with_neuroexapt(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=50,
        learning_rate=0.001,
        efficiency_threshold=0.05,
        verbose=True
    )
    
    print(f"\n📊 Training Results:")
    print(f"  Final Train Acc: {history['train_accuracy'][-1]:.1f}% | Final Val Acc: {history['val_accuracy'][-1]:.1f}%")
    print(f"  Best Val Acc: {max(history['val_accuracy']):.1f}% | Evolutions: {sum(history['evolutions'])}")
    
    # Check for overfitting
    final_train_acc = history['train_accuracy'][-1]
    final_val_acc = history['val_accuracy'][-1]
    overfitting_gap = final_train_acc - final_val_acc
    
    print(f"\n🔍 Overfitting Analysis:")
    print(f"  Train-Val gap: {overfitting_gap:.1f}%")
    if overfitting_gap < 10:
        print("  ✅ Good generalization - minimal overfitting")
    elif overfitting_gap < 20:
        print("  ⚠️  Moderate overfitting - acceptable")
    else:
        print("  ❌ High overfitting - model needs improvement")
    
    print("\n🔍 Architecture Analysis:")
    
    # Create another model for comparison
    model2 = ImprovedCNN(num_classes=10)
    trainer = Trainer(
        model=model2,
        device=device,
        efficiency_threshold=0.1,
        verbose=True
    )
    
    # Architecture analysis
    print("📋 Architecture Analysis:")
    analysis = trainer.analyze_architecture(val_loader)
    print(f"  Efficiency: {analysis['computational_efficiency']:.3f}")
    print(f"  Redundancy: {analysis['total_redundancy']:.3f}")
    print(f"  Conv layers: {analysis['conv_layers']}")
    print(f"  Linear layers: {analysis['linear_layers']}")
    
    # Train with detailed tracking (shorter for demo)
    print("\n🎯 Quick training demonstration (10 epochs):")
    history2 = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=10,
        learning_rate=0.001,
        optimizer_type='adam'
    )
    
    print(f"\n📊 Quick Training Results:")
    print(f"  Final train accuracy: {history2['train_accuracy'][-1]:.1f}%")
    print(f"  Final val accuracy: {history2['val_accuracy'][-1]:.1f}%")
    print(f"  Evolutions: {sum(history2['evolutions'])}")
    
    # Evolution summary
    evolution_summary = trainer.get_evolution_summary()
    print(f"\n🧠 Evolution Intelligence:")
    print(f"  Success rate: {evolution_summary['evolution_stats']['success_rate']:.1%}")
    print(f"  No-change rate: {evolution_summary['evolution_stats']['no_change_rate']:.1%}")
    print(f"  Performance trend: {evolution_summary['performance_trend']}")
    
    print("\n✅ Training completed with intelligent architecture evolution!")


if __name__ == "__main__":
    main()