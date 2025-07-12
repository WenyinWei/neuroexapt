#!/usr/bin/env python3
"""
Quick Test for NeuroExapt V3

This script performs a quick validation of NeuroExapt V3 functionality.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings
import sys

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

print("🧠 NeuroExapt V3 - Quick Test")
print("=" * 50)

# Test 1: Import V3 modules
print("1. Testing imports...")
try:
    from neuroexapt.trainer_v3 import TrainerV3, train_with_neuroexapt
    from neuroexapt.neuroexapt_v3 import NeuroExaptV3
    print("   ✓ All V3 modules imported successfully")
except ImportError as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Create simple model
print("2. Creating test model...")
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 16, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = SimpleNet()
print(f"   ✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

# Test 3: Create dummy data
print("3. Creating dummy data...")
def create_dummy_data(n_samples=200):
    X = torch.randn(n_samples, 3, 16, 16)
    y = torch.randint(0, 10, (n_samples,))
    return TensorDataset(X, y)

train_dataset = create_dummy_data(200)
val_dataset = create_dummy_data(50)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

print(f"   ✓ Dummy data created: {len(train_dataset)} train, {len(val_dataset)} val")

# Test 4: Test TrainerV3 initialization
print("4. Testing TrainerV3 initialization...")
try:
    trainer = TrainerV3(
        model=model,
        efficiency_threshold=0.05,
        verbose=False
    )
    print("   ✓ TrainerV3 initialized successfully")
except Exception as e:
    print(f"   ✗ TrainerV3 initialization failed: {e}")
    sys.exit(1)

# Test 5: Test architecture analysis
print("5. Testing architecture analysis...")
try:
    analysis = trainer.analyze_architecture(val_loader)
    print(f"   ✓ Architecture analysis completed")
    print(f"     - Total redundancy: {analysis['total_redundancy']:.3f}")
    print(f"     - Computational efficiency: {analysis['computational_efficiency']:.3f}")
    print(f"     - Total parameters: {analysis['total_parameters']:,}")
except Exception as e:
    print(f"   ✗ Architecture analysis failed: {e}")
    sys.exit(1)

# Test 6: Test short training
print("6. Testing short training...")
try:
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=3,  # Very short for testing
        learning_rate=0.01,
        optimizer_type='adam'
    )
    print(f"   ✓ Short training completed")
    print(f"     - Final train accuracy: {history['train_accuracy'][-1]:.1f}%")
    print(f"     - Final val accuracy: {history['val_accuracy'][-1]:.1f}%")
    print(f"     - Total evolutions: {sum(history['evolutions'])}")
except Exception as e:
    print(f"   ✗ Training failed: {e}")
    sys.exit(1)

# Test 7: Test convenience function
print("7. Testing convenience function...")
try:
    model2 = SimpleNet()
    optimized_model, history2 = train_with_neuroexapt(
        model=model2,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=2,
        learning_rate=0.01,
        efficiency_threshold=0.05,
        verbose=False
    )
    print(f"   ✓ Convenience function completed")
    print(f"     - Final accuracy: {history2['val_accuracy'][-1]:.1f}%")
    print(f"     - Total evolutions: {sum(history2['evolutions'])}")
except Exception as e:
    print(f"   ✗ Convenience function failed: {e}")
    sys.exit(1)

# Final summary
print("\n" + "=" * 50)
print("🎉 All tests passed! NeuroExapt V3 is working correctly!")
print("=" * 50)

print("\n📊 Quick Test Results:")
print("✓ Module imports - PASSED")
print("✓ Model creation - PASSED")
print("✓ Data loading - PASSED")
print("✓ Trainer initialization - PASSED")
print("✓ Architecture analysis - PASSED")
print("✓ Training process - PASSED")
print("✓ Convenience function - PASSED")

print("\n🚀 NeuroExapt V3 is ready for use!")
print("\n✨ Quick test completed successfully!")