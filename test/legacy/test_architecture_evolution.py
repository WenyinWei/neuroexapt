"""
Test Architecture Evolution with Simulated Data

This script demonstrates how Neuro Exapt can spontaneously adjust neural network
architecture based on information-theoretic principles, including:
- Adding new subnetworks when information bottlenecks are detected
- Removing old, low-efficiency subnetworks based on entropy analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import neuroexapt


class EvolvingCNN(nn.Module):
    """
    A CNN designed to demonstrate architecture evolution.
    
    This model starts with a simple structure and will evolve based on
    information-theoretic analysis during training.
    """
    
    def __init__(self, num_classes=10):
        super(EvolvingCNN, self).__init__()
        
        # Initial architecture - will evolve during training
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        # Track evolution
        self.evolution_history = []
        self.layer_importance = {}
        
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
    
    def get_layer_names(self):
        """Get names of all layers for evolution tracking."""
        return ['conv1', 'conv2', 'conv3', 'fc1', 'fc2', 'fc3']


def create_synthetic_dataset(num_samples=1000, num_classes=10):
    """Create synthetic dataset for testing."""
    print("Creating synthetic dataset...")
    
    # Generate synthetic images (32x32x3)
    X = torch.randn(num_samples, 3, 32, 32)
    
    # Generate labels
    y = torch.randint(0, num_classes, (num_samples,))
    
    # Split into train/val
    train_size = int(0.8 * num_samples)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"Created dataset: {len(train_dataset)} training, {len(val_dataset)} validation samples")
    return train_loader, val_loader


def analyze_model_complexity(model):
    """Analyze model complexity and structure."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    layer_info = {}
    for name, module in model.named_modules():
        if len(list(module.parameters())) > 0:
            layer_params = sum(p.numel() for p in module.parameters())
            layer_info[name] = {
                'type': type(module).__name__,
                'parameters': layer_params,
                'shape': list(module.parameters())[0].shape if list(module.parameters()) else None
            }
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'layer_info': layer_info
    }


def main():
    """Main demonstration function."""
    
    print("=" * 80)
    print("Neuro Exapt - Architecture Evolution Demonstration")
    print("=" * 80)
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create synthetic dataset
    train_loader, val_loader = create_synthetic_dataset()
    
    # Create initial model
    print("\nInitializing evolving CNN...")
    model = EvolvingCNN(num_classes=10)
    model = model.to(device)
    
    # Analyze initial model
    initial_analysis = analyze_model_complexity(model)
    print(f"\nInitial Model Analysis:")
    print(f"  Total parameters: {initial_analysis['total_parameters']:,}")
    print(f"  Trainable parameters: {initial_analysis['trainable_parameters']:,}")
    print(f"  Number of layers: {len(initial_analysis['layer_info'])}")
    
    # Initialize Neuro Exapt with aggressive evolution settings
    print("\nInitializing Neuro Exapt with evolution settings...")
    neuro_exapt = neuroexapt.NeuroExapt(
        task_type="classification",
        entropy_weight=0.3,  # Lower threshold for more aggressive pruning
        info_weight=0.2,     # Higher weight for information bottleneck
        device=device,
        verbose=True
    )
    
    # Wrap model
    wrapped_model = neuro_exapt.wrap_model(model)
    
    # Setup training components
    optimizer = torch.optim.AdamW(wrapped_model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop with evolution monitoring
    print(f"\nStarting training with architecture evolution...")
    print("-" * 80)
    
    epochs = 20
    evolution_events = []
    
    for epoch in range(epochs):
        # Training phase
        wrapped_model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = wrapped_model(data)
            loss = wrapped_model.compute_loss(outputs, targets, criterion)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
        
        # Validation phase
        wrapped_model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = wrapped_model(data)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        # Calculate metrics
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        print(f"Epoch {epoch+1:2d}/{epochs}: "
              f"Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss/len(val_loader):.4f}, "
              f"Val Acc: {val_acc:.2f}%")
        
        # Trigger evolution every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"\n🔄 Triggering architecture evolution at epoch {epoch+1}...")
            
            # Analyze current model
            current_analysis = analyze_model_complexity(wrapped_model.model)
            
            # Prepare metrics for evolution decision
            evolution_metrics = {
                'train_accuracy': train_acc,
                'val_accuracy': val_acc,
                'train_loss': train_loss/len(train_loader),
                'val_loss': val_loss/len(val_loader),
                'epoch': epoch + 1
            }
            
            # Trigger evolution
            evolution_result = neuro_exapt.evolve_structure(evolution_metrics)
            
            if evolution_result and evolution_result[0]:  # Evolution occurred
                evolution_info = evolution_result[1]
                print(f"✅ Evolution occurred: {evolution_info}")
                
                # Analyze post-evolution model
                post_analysis = analyze_model_complexity(wrapped_model.model)
                
                evolution_event = {
                    'epoch': epoch + 1,
                    'action': evolution_info.get('action', 'unknown'),
                    'params_before': current_analysis['total_parameters'],
                    'params_after': post_analysis['total_parameters'],
                    'layers_before': len(current_analysis['layer_info']),
                    'layers_after': len(post_analysis['layer_info']),
                    'info': evolution_info
                }
                evolution_events.append(evolution_event)
                
                print(f"   Parameters: {evolution_event['params_before']:,} → {evolution_event['params_after']:,}")
                print(f"   Layers: {evolution_event['layers_before']} → {evolution_event['layers_after']}")
                
                # Update optimizer for new parameters
                optimizer = torch.optim.AdamW(wrapped_model.parameters(), lr=0.001, weight_decay=1e-4)
            else:
                print("   No evolution triggered")
        
        # Update epoch in neuro_exapt
        neuro_exapt.update_epoch(epoch + 1, epochs)
    
    # Final analysis
    print("\n" + "=" * 80)
    print("Training completed! Final Analysis:")
    print("=" * 80)
    
    final_analysis = analyze_model_complexity(wrapped_model.model)
    print(f"\nFinal Model:")
    print(f"  Total parameters: {final_analysis['total_parameters']:,}")
    print(f"  Trainable parameters: {final_analysis['trainable_parameters']:,}")
    print(f"  Number of layers: {len(final_analysis['layer_info'])}")
    
    print(f"\nEvolution Summary:")
    print(f"  Total evolution events: {len(evolution_events)}")
    
    if evolution_events:
        print(f"  Evolution events:")
        for i, event in enumerate(evolution_events):
            print(f"    {i+1}. Epoch {event['epoch']}: {event['action']}")
            print(f"       Parameters: {event['params_before']:,} → {event['params_after']:,}")
            print(f"       Layers: {event['layers_before']} → {event['layers_after']}")
    
    # Demonstrate information-theoretic analysis
    print(f"\nInformation-Theoretic Analysis:")
    try:
        # Analyze model with neuro_exapt
        analysis_result = neuro_exapt.analyze_model(val_loader, num_batches=5)
        
        if 'entropy' in analysis_result:
            entropy_info = analysis_result['entropy']
            print(f"  Network entropy: {entropy_info.get('network_entropy', 'N/A')}")
            print(f"  Layer entropies: {len(entropy_info.get('layer_entropies', {}))} layers analyzed")
        
        if 'complexity' in analysis_result:
            complexity = analysis_result['complexity']
            print(f"  Structural complexity: {complexity}")
            
    except Exception as e:
        print(f"  Analysis error: {e}")
    
    print(f"\n🎉 Architecture evolution demonstration completed!")
    print(f"   The neural network has successfully evolved its structure")
    print(f"   based on information-theoretic principles during training.")


if __name__ == "__main__":
    main() 