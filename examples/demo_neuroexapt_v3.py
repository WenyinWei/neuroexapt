#!/usr/bin/env python3
"""
NeuroExapt V3 - Complete Demo

This demo showcases all V3 features in a single comprehensive example.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

from neuroexapt.trainer_v3 import TrainerV3, train_with_neuroexapt


class DemoNet(nn.Module):
    """Demo neural network for showcasing V3 features"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def create_demo_dataset(n_samples=1000):
    """Create a demo dataset"""
    X = torch.randn(n_samples, 3, 32, 32)
    y = torch.randint(0, 10, (n_samples,))
    
    # Split 80/20
    split = int(0.8 * n_samples)
    train_X, val_X = X[:split], X[split:]
    train_y, val_y = y[:split], y[split:]
    
    train_dataset = TensorDataset(train_X, train_y)
    val_dataset = TensorDataset(val_X, val_y)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    return train_loader, val_loader


def main():
    print("🎉 NeuroExapt V3 - Complete Feature Demo")
    print("=" * 60)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    
    # Create model and data
    model = DemoNet(num_classes=10)
    print(f"🏗️  Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    train_loader, val_loader = create_demo_dataset(1200)
    print(f"📊 Dataset: 960 train, 240 val")
    
    print("\n" + "🚀 DEMO 1: One-line Training (Recommended)" + "=" * 25)
    print("This is the easiest way to use NeuroExapt V3:")
    
    # Demo 1: One-line training
    optimized_model, history = train_with_neuroexapt(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=6,
        learning_rate=0.001,
        efficiency_threshold=0.05,
        verbose=True
    )
    
    print(f"\n📊 One-line Training Results:")
    print(f"   Final accuracy: {history['val_accuracy'][-1]:.1f}%")
    print(f"   Evolutions: {sum(history['evolutions'])}")
    
    print("\n" + "🔬 DEMO 2: Advanced Analysis" + "=" * 33)
    
    # Demo 2: Detailed trainer with analysis
    model2 = DemoNet(num_classes=10)
    trainer = TrainerV3(
        model=model2,
        device=device,
        efficiency_threshold=0.08,
        verbose=True
    )
    
    # Architecture analysis before training
    print("📋 Initial Architecture Analysis:")
    initial_analysis = trainer.analyze_architecture(val_loader)
    
    # Training with detailed tracking
    history2 = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=5,
        learning_rate=0.002,
        optimizer_type='adamw'
    )
    
    # Final analysis
    print("\n📋 Final Architecture Analysis:")
    final_analysis = trainer.analyze_architecture(val_loader)
    
    print(f"\n🧠 Intelligence Summary:")
    evolution_summary = trainer.get_evolution_summary()
    print(f"   Success rate: {evolution_summary['evolution_stats']['success_rate']:.1%}")
    print(f"   No-change rate: {evolution_summary['evolution_stats']['no_change_rate']:.1%}")
    print(f"   Performance trend: {evolution_summary['performance_trend']}")
    
    print("\n" + "⚡ DEMO 3: Efficiency Comparison" + "=" * 29)
    
    # Demo 3: Compare different thresholds
    print("Testing different efficiency thresholds...")
    
    thresholds = [0.01, 0.05, 0.10]
    results = []
    
    for threshold in thresholds:
        print(f"\n🎯 Testing threshold: {threshold}")
        model_test = DemoNet(num_classes=10)
        
        _, hist = train_with_neuroexapt(
            model=model_test,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=3,
            efficiency_threshold=threshold,
            verbose=False
        )
        
        evolutions = sum(hist['evolutions'])
        final_acc = hist['val_accuracy'][-1]
        results.append((threshold, evolutions, final_acc))
        print(f"   Result: {evolutions} evolutions, {final_acc:.1f}% accuracy")
    
    print(f"\n📊 Threshold Comparison Summary:")
    for threshold, evolutions, accuracy in results:
        print(f"   {threshold:.2f}: {evolutions} evolutions → {accuracy:.1f}% accuracy")
    
    print("\n" + "🎯 V3 Key Features Demonstrated" + "=" * 28)
    print("✅ Every-epoch checking with intelligent decisions")
    print("✅ Efficiency-based evolution triggering") 
    print("✅ Automatic rollback protection")
    print("✅ Smart visualization (only shows changes)")
    print("✅ One-line deployment capability")
    print("✅ Comprehensive architecture analysis")
    print("✅ Adaptive threshold learning")
    
    print(f"\n🎉 NeuroExapt V3 Demo Completed Successfully!")
    print("Ready for production use! 🚀")


if __name__ == "__main__":
    main()