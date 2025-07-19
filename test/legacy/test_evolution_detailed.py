"""
Test Detailed Architecture Evolution

This script tests the improved architecture evolution with detailed reporting
and more aggressive layer addition for better accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import neuroexapt
from neuroexapt.trainer import Trainer


class TestCNN(nn.Module):
    """A test CNN that can be easily expanded."""
    
    def __init__(self, num_classes=10):
        super(TestCNN, self).__init__()
        
        # Initial layers
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
        
        # Track if we've added expansion layers
        self.expansion_layers = []
        
    def forward(self, x):
        # Convolutional layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Apply expansion layers if any
        for layer_name in self.expansion_layers:
            if hasattr(self, layer_name):
                layer = getattr(self, layer_name)
                x = F.relu(layer(x))
        
        # Flatten
        x = x.view(-1, 128 * 4 * 4)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
    def add_expansion_layer(self, layer_name, in_channels=128, out_channels=128, device=None):
        """Add a new expansion layer."""
        if layer_name not in self.expansion_layers:
            new_layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            if device is not None:
                new_layer = new_layer.to(device)
            setattr(self, layer_name, new_layer)
            self.expansion_layers.append(layer_name)
            print(f"✅ Added expansion layer: {layer_name}")
            return True
        return False
    
    def get_layer_info(self):
        """Get information about current layers."""
        return {
            'base_layers': ['conv1', 'conv2', 'conv3', 'fc1', 'fc2', 'fc3'],
            'expansion_layers': self.expansion_layers,
            'total_layers': len(self.expansion_layers) + 6
        }


def create_simple_dataset():
    """Create a simple dataset for testing."""
    print("Creating simple test dataset...")
    
    # Generate synthetic data
    X = torch.randn(1000, 3, 32, 32)
    y = torch.randint(0, 10, (1000,))
    
    # Split
    train_size = 800
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    # Create datasets
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"Created dataset: {len(train_dataset)} training, {len(val_dataset)} validation samples")
    return train_loader, val_loader


def main():
    """Main test function."""
    
    print("=" * 80)
    print("Test Detailed Architecture Evolution")
    print("=" * 80)
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataset
    train_loader, val_loader = create_simple_dataset()
    
    # Create model
    print("\nInitializing test CNN...")
    model = TestCNN(num_classes=10)
    model = model.to(device)
    
    # Analyze initial model
    initial_params = sum(p.numel() for p in model.parameters())
    layer_info = model.get_layer_info()
    print(f"\nInitial Model:")
    print(f"  Parameters: {initial_params:,}")
    print(f"  Layers: {layer_info['total_layers']}")
    print(f"  Base layers: {len(layer_info['base_layers'])}")
    print(f"  Expansion layers: {len(layer_info['expansion_layers'])}")
    
    # Initialize Neuro Exapt with aggressive settings
    print("\nInitializing Neuro Exapt with aggressive evolution...")
    neuro_exapt_instance = neuroexapt.NeuroExapt(
        task_type="classification",
        entropy_weight=0.2,  # Lower threshold for more aggressive evolution
        info_weight=0.3,     # Higher weight for information bottleneck
        device=device,
        verbose=True
    )
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        neuro_exapt=neuro_exapt_instance,
        evolution_frequency=3,  # More frequent evolution
        device=device,
        verbose=True
    )
    
    # Training with detailed evolution monitoring
    print(f"\nStarting training with detailed evolution monitoring...")
    print("-" * 80)
    
    epochs = 20
    evolution_events = []
    
    for epoch in range(epochs):
        # Training phase
        trainer.wrapped_model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            trainer.optimizer.zero_grad()
            outputs = trainer.wrapped_model(data)
            loss = trainer.wrapped_model.compute_loss(outputs, targets, nn.CrossEntropyLoss())
            loss.backward()
            trainer.optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
        
        # Validation phase
        trainer.wrapped_model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = trainer.wrapped_model(data)
                loss = nn.CrossEntropyLoss()(outputs, targets)
                
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
        
        # Check for evolution every 3 epochs
        if (epoch + 1) % 3 == 0:
            print(f"\n🔄 Checking for evolution at epoch {epoch+1}...")
            
            # Get current model info
            current_params = sum(p.numel() for p in trainer.wrapped_model.parameters())
            current_layer_info = trainer.wrapped_model.model.get_layer_info()
            
            # Prepare metrics for evolution
            evolution_metrics = {
                'train_accuracy': train_acc,
                'val_accuracy': val_acc,
                'train_loss': train_loss/len(train_loader),
                'val_loss': val_loss/len(val_loader),
                'epoch': epoch + 1
            }
            
            # Trigger evolution
            evolution_result = neuro_exapt_instance.evolve_structure(evolution_metrics)
            
            if evolution_result and evolution_result[0]:  # Evolution occurred
                evolution_info = evolution_result[1]
                action = evolution_info.get('action', 'unknown')
                
                                 # Manual layer addition for testing
                 if action == 'expand' and val_acc < 85.0:
                     expansion_count = len(current_layer_info['expansion_layers'])
                     new_layer_name = f'expansion_conv_{expansion_count + 1}'
                     trainer.wrapped_model.model.add_expansion_layer(new_layer_name, device=device)
                    
                    # Update optimizer for new parameters
                    trainer.optimizer = torch.optim.AdamW(
                        trainer.wrapped_model.parameters(),
                        lr=0.001,
                        weight_decay=1e-4
                    )
                    
                    # Record evolution event
                    evolution_events.append({
                        'epoch': epoch + 1,
                        'action': 'expand',
                        'layer_added': new_layer_name,
                        'params_before': current_params,
                        'params_after': sum(p.numel() for p in trainer.wrapped_model.parameters()),
                        'accuracy_before': val_acc,
                        'accuracy_after': val_acc  # Will be updated next epoch
                    })
                    
                    print(f"✅ Evolution occurred: Added layer {new_layer_name}")
                    print(f"   Parameters: {current_params:,} → {sum(p.numel() for p in trainer.wrapped_model.parameters()):,}")
                else:
                    print(f"   No evolution triggered (action: {action})")
            else:
                print(f"   No evolution triggered")
        
        # Update epoch in neuro_exapt
        neuro_exapt_instance.update_epoch(epoch + 1, epochs)
    
    # Final analysis
    print("\n" + "=" * 80)
    print("Training completed! Final Analysis:")
    print("=" * 80)
    
    final_params = sum(p.numel() for p in trainer.wrapped_model.parameters())
    final_layer_info = trainer.wrapped_model.model.get_layer_info()
    
    print(f"\nFinal Model:")
    print(f"  Parameters: {final_params:,}")
    print(f"  Total layers: {final_layer_info['total_layers']}")
    print(f"  Expansion layers: {final_layer_info['expansion_layers']}")
    
    print(f"\nEvolution Summary:")
    print(f"  Total evolution events: {len(evolution_events)}")
    
    if evolution_events:
        print(f"  Evolution events:")
        for i, event in enumerate(evolution_events):
            print(f"    {i+1}. Epoch {event['epoch']}: {event['action']}")
            print(f"       Layer added: {event['layer_added']}")
            print(f"       Parameters: {event['params_before']:,} → {event['params_after']:,}")
            param_change = event['params_after'] - event['params_before']
            print(f"       Change: +{param_change:,} parameters")
    
    print(f"\n🎉 Detailed evolution test completed!")
    print(f"   The model has evolved with detailed reporting of structural changes.")


if __name__ == "__main__":
    main() 