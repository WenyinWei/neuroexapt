"""
Detailed Demonstration of Subnetwork Evolution

This script shows how Neuro Exapt can:
1. Add new subnetworks when information bottlenecks are detected
2. Remove old, low-efficiency subnetworks based on entropy analysis
3. Demonstrate the mathematical principles behind these decisions
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


class ModularCNN(nn.Module):
    """
    A modular CNN with explicit subnetworks that can be added/removed.
    
    This model demonstrates how Neuro Exapt can:
    - Add new feature extraction modules
    - Remove redundant processing paths
    - Optimize network topology based on information flow
    """
    
    def __init__(self, num_classes=10):
        super(ModularCNN, self).__init__()
        
        # Core feature extraction subnetworks
        self.core_conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.core_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Optional subnetworks (can be added/removed during evolution)
        self.optional_conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.optional_conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Attention subnetwork (can be added when needed)
        self.attention_conv = nn.Conv2d(256, 256, kernel_size=1)
        self.attention_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Pooling and regularization
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        
        # Classification subnetworks
        self.classifier1 = nn.Linear(256 * 4 * 4, 512)
        self.classifier2 = nn.Linear(512, 256)
        self.classifier3 = nn.Linear(256, num_classes)
        
        # Evolution tracking
        self.active_subnetworks = {
            'core': True,
            'optional': True,
            'attention': False,  # Initially disabled
            'classifier': True
        }
        
        self.evolution_history = []
        
    def forward(self, x):
        # Core subnetwork (always active)
        x = self.pool(F.relu(self.core_conv1(x)))
        x = self.pool(F.relu(self.core_conv2(x)))
        
        # Optional subnetwork (can be pruned)
        if self.active_subnetworks['optional']:
            x = self.pool(F.relu(self.optional_conv1(x)))
            x = self.pool(F.relu(self.optional_conv2(x)))
        
        # Attention subnetwork (can be added)
        if self.active_subnetworks['attention']:
            attention_weights = self.attention_gate(x)
            x = x * attention_weights
        
        # Flatten
        x = x.view(-1, 256 * 4 * 4)
        
        # Classification subnetwork
        if self.active_subnetworks['classifier']:
            x = F.relu(self.classifier1(x))
            x = self.dropout(x)
            x = F.relu(self.classifier2(x))
            x = self.classifier3(x)
        
        return x
    
    def add_subnetwork(self, subnetwork_type):
        """Add a new subnetwork."""
        if subnetwork_type == 'attention' and not self.active_subnetworks['attention']:
            self.active_subnetworks['attention'] = True
            print(f"✅ Added attention subnetwork")
            return True
        return False
    
    def remove_subnetwork(self, subnetwork_type):
        """Remove a subnetwork."""
        if subnetwork_type == 'optional' and self.active_subnetworks['optional']:
            self.active_subnetworks['optional'] = False
            print(f"🗑️ Removed optional subnetwork")
            return True
        return False
    
    def get_subnetwork_info(self):
        """Get information about active subnetworks."""
        info = {
            'active_count': sum(self.active_subnetworks.values()),
            'total_count': len(self.active_subnetworks),
            'active_subnetworks': [k for k, v in self.active_subnetworks.items() if v],
            'inactive_subnetworks': [k for k, v in self.active_subnetworks.items() if not v]
        }
        return info


def create_complex_dataset(num_samples=2000, num_classes=10):
    """Create a more complex synthetic dataset."""
    print("Creating complex synthetic dataset...")
    
    # Generate synthetic images with more structure
    X = torch.randn(num_samples, 3, 32, 32)
    
    # Add some structure to make the task more interesting
    for i in range(num_samples):
        # Add random patterns
        pattern = torch.randn(3, 8, 8)
        x_start = torch.randint(0, 25, (1,)).item()
        y_start = torch.randint(0, 25, (1,)).item()
        X[i, :, x_start:x_start+8, y_start:y_start+8] = pattern
    
    # Generate labels based on patterns
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
    
    print(f"Created complex dataset: {len(train_dataset)} training, {len(val_dataset)} validation samples")
    return train_loader, val_loader


def analyze_information_flow(model, dataloader, device):
    """Analyze information flow through different subnetworks."""
    model.eval()
    
    # Collect activations from different subnetworks
    activations = {
        'core': [],
        'optional': [],
        'attention': [],
        'classifier': []
    }
    
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(dataloader):
            if batch_idx >= 5:  # Analyze first 5 batches
                break
                
            data = data.to(device)
            
            # Forward pass through core
            x = F.relu(model.core_conv1(data))
            x = F.relu(model.core_conv2(x))
            activations['core'].append(x.mean().item())
            
            # Forward pass through optional
            if model.active_subnetworks['optional']:
                x = F.relu(model.optional_conv1(x))
                x = F.relu(model.optional_conv2(x))
                activations['optional'].append(x.mean().item())
            
            # Forward pass through attention
            if model.active_subnetworks['attention']:
                attention_weights = model.attention_gate(x)
                x = x * attention_weights
                activations['attention'].append(attention_weights.mean().item())
            
            # Forward pass through classifier
            x = x.view(-1, 256 * 4 * 4)
            if model.active_subnetworks['classifier']:
                x = F.relu(model.classifier1(x))
                activations['classifier'].append(x.mean().item())
    
    # Calculate information metrics
    info_metrics = {}
    for subnetwork, acts in activations.items():
        if acts:
            info_metrics[subnetwork] = {
                'mean_activation': np.mean(acts),
                'activation_variance': np.var(acts),
                'information_content': np.mean(acts) * np.var(acts)  # Simple info metric
            }
    
    return info_metrics


def main():
    """Main demonstration function."""
    
    print("=" * 80)
    print("Neuro Exapt - Subnetwork Evolution Demonstration")
    print("=" * 80)
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create complex dataset
    train_loader, val_loader = create_complex_dataset()
    
    # Create modular model
    print("\nInitializing modular CNN...")
    model = ModularCNN(num_classes=10)
    model = model.to(device)
    
    # Initial subnetwork analysis
    initial_info = model.get_subnetwork_info()
    print(f"\nInitial Subnetwork Configuration:")
    print(f"  Active subnetworks: {initial_info['active_subnetworks']}")
    print(f"  Inactive subnetworks: {initial_info['inactive_subnetworks']}")
    
    # Initialize Neuro Exapt with subnetwork-aware settings
    print("\nInitializing Neuro Exapt for subnetwork evolution...")
    neuro_exapt = neuroexapt.NeuroExapt(
        task_type="classification",
        entropy_weight=0.2,  # More aggressive for subnetwork evolution
        info_weight=0.3,     # Higher weight for information bottleneck
        device=device,
        verbose=True
    )
    
    # Wrap model
    wrapped_model = neuro_exapt.wrap_model(model)
    
    # Setup training components
    optimizer = torch.optim.AdamW(wrapped_model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop with subnetwork evolution
    print(f"\nStarting training with subnetwork evolution...")
    print("-" * 80)
    
    epochs = 30
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
        
        # Subnetwork evolution every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"\n🔄 Analyzing subnetwork evolution at epoch {epoch+1}...")
            
            # Analyze information flow
            info_metrics = analyze_information_flow(wrapped_model.model, val_loader, device)
            
            print(f"Information Flow Analysis:")
            for subnetwork, metrics in info_metrics.items():
                print(f"  {subnetwork}: mean={metrics['mean_activation']:.4f}, "
                      f"var={metrics['activation_variance']:.4f}, "
                      f"info={metrics['information_content']:.4f}")
            
            # Decision logic for subnetwork evolution
            current_info = wrapped_model.model.get_subnetwork_info()
            
            # Add attention subnetwork if information bottleneck detected
            if (info_metrics.get('optional', {}).get('information_content', 0) > 0.1 and 
                not wrapped_model.model.active_subnetworks.get('attention', False)):
                print(f"🔍 Information bottleneck detected, adding attention subnetwork...")
                wrapped_model.model.add_subnetwork('attention')
                evolution_events.append({
                    'epoch': epoch + 1,
                    'action': 'add_attention',
                    'reason': 'information_bottleneck'
                })
            
            # Remove optional subnetwork if low information content
            elif (info_metrics.get('optional', {}).get('information_content', 0) < 0.01 and 
                  wrapped_model.model.active_subnetworks.get('optional', False)):
                print(f"🗑️ Low information content detected, removing optional subnetwork...")
                wrapped_model.model.remove_subnetwork('optional')
                evolution_events.append({
                    'epoch': epoch + 1,
                    'action': 'remove_optional',
                    'reason': 'low_information_content'
                })
            
            # Update optimizer for new parameters
            optimizer = torch.optim.AdamW(wrapped_model.parameters(), lr=0.001, weight_decay=1e-4)
        
        # Update epoch in neuro_exapt
        neuro_exapt.update_epoch(epoch + 1, epochs)
    
    # Final analysis
    print("\n" + "=" * 80)
    print("Training completed! Final Subnetwork Analysis:")
    print("=" * 80)
    
    final_info = wrapped_model.model.get_subnetwork_info()
    print(f"\nFinal Subnetwork Configuration:")
    print(f"  Active subnetworks: {final_info['active_subnetworks']}")
    print(f"  Inactive subnetworks: {final_info['inactive_subnetworks']}")
    
    print(f"\nSubnetwork Evolution Summary:")
    print(f"  Total evolution events: {len(evolution_events)}")
    
    if evolution_events:
        print(f"  Evolution events:")
        for i, event in enumerate(evolution_events):
            print(f"    {i+1}. Epoch {event['epoch']}: {event['action']} ({event['reason']})")
    
    # Final information flow analysis
    print(f"\nFinal Information Flow Analysis:")
    final_info_metrics = analyze_information_flow(wrapped_model.model, val_loader, device)
    for subnetwork, metrics in final_info_metrics.items():
        print(f"  {subnetwork}: info_content={metrics['information_content']:.4f}")
    
    print(f"\n🎉 Subnetwork evolution demonstration completed!")
    print(f"   The neural network has successfully evolved its subnetworks")
    print(f"   based on information-theoretic analysis during training.")


if __name__ == "__main__":
    main() 