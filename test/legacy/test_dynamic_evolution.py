"""
Test script to verify dynamic architecture evolution improvements.

This script creates a simple scenario to test if the evolution triggers properly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuroexapt.neuroexapt import NeuroExapt
from neuroexapt.trainer import Trainer


class SimpleTestNet(nn.Module):
    """Simple network for testing evolution."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def create_overfitting_dataset():
    """Create a dataset that will cause overfitting."""
    # Small training set
    X_train = torch.randn(100, 10)
    y_train = torch.randint(0, 2, (100,))
    
    # Larger test set with different distribution
    X_test = torch.randn(500, 10) + 0.5  # Shift distribution
    y_test = torch.randint(0, 2, (500,))
    
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    return DataLoader(train_dataset, batch_size=10, shuffle=True), \
           DataLoader(test_dataset, batch_size=50)


def main():
    print("=" * 60)
    print("🧪 Testing Dynamic Architecture Evolution")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # Create model
    model = SimpleTestNet()
    print(f"Initial parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Initialize NeuroExapt
    neuro_exapt = NeuroExapt(
        task_type="classification",
        entropy_weight=0.3,
        info_weight=0.1,
        device=device,
        verbose=True
    )
    
    # Wrap model
    wrapped_model = neuro_exapt.wrap_model(model)
    
    # Create optimizer
    optimizer = torch.optim.Adam(wrapped_model.parameters(), lr=0.01)
    
    # Create trainer with aggressive evolution settings
    trainer = Trainer(
        model=wrapped_model,
        neuro_exapt=neuro_exapt,
        optimizer=optimizer,
        evolution_frequency=5,  # Check frequently
        device=device,
        verbose=True
    )
    
    # Manually adjust thresholds for testing
    trainer.min_epochs_between_evolution = 2  # Allow frequent evolution
    trainer.entropy_variance_threshold = 0.1  # Very relaxed
    
    # Create overfitting dataset
    train_loader, test_loader = create_overfitting_dataset()
    
    print("\n🚀 Starting training with overfitting scenario...")
    print("This should trigger evolution due to train/val gap\n")
    
    # Track evolution events
    evolution_count = 0
    
    # Manual training loop for better control
    for epoch in range(20):
        # Update epoch
        neuro_exapt.update_epoch(epoch, 20)
        
        # Train
        wrapped_model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = wrapped_model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += pred.eq(target).sum().item()
            train_total += target.size(0)
        
        train_acc = 100. * train_correct / train_total
        
        # Validate
        wrapped_model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = wrapped_model(data)
                pred = output.argmax(dim=1)
                val_correct += pred.eq(target).sum().item()
                val_total += target.size(0)
        
        val_acc = 100. * val_correct / val_total
        
        # Calculate entropy
        with torch.no_grad():
            sample_data = next(iter(train_loader))[0].to(device)
            sample_output = wrapped_model(sample_data)
            current_entropy = neuro_exapt.entropy_ctrl.measure(sample_output)
        
        print(f"Epoch {epoch+1:2d} | Train: {train_acc:5.1f}% | Val: {val_acc:5.1f}% | "
              f"Gap: {train_acc-val_acc:5.1f}% | Entropy: {current_entropy:.3f}")
        
        # Check if evolution should trigger
        train_metrics = {'accuracy': train_acc, 'val_accuracy': val_acc}
        if trainer._should_evolve_structure(epoch, train_metrics):
            print("\n✅ Evolution triggered!")
            
            # Perform evolution
            performance_metrics = {
                'accuracy': train_acc,
                'val_accuracy': val_acc,
                'loss': train_loss / len(train_loader)
            }
            
            evolved, info = neuro_exapt.evolve_structure(performance_metrics)
            
            if evolved:
                evolution_count += 1
                new_params = sum(p.numel() for p in wrapped_model.parameters())
                print(f"   Action: {info['action']}")
                print(f"   Parameters: {new_params}")
                
                # Update optimizer
                optimizer = torch.optim.Adam(wrapped_model.parameters(), lr=0.01)
                trainer.optimizer = optimizer
            print()
    
    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    print(f"   Total epochs: 20")
    print(f"   Evolution events: {evolution_count}")
    print(f"   Final parameters: {sum(p.numel() for p in wrapped_model.parameters())}")
    
    if evolution_count > 0:
        print("\n✅ SUCCESS: Dynamic evolution is working!")
    else:
        print("\n⚠️  WARNING: No evolution triggered. Check thresholds.")
    
    print("=" * 60)


if __name__ == "__main__":
    main() 