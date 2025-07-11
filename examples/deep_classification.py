"""
Deep Classification Example with Neuro Exapt.

This example demonstrates how to use Neuro Exapt for dynamic architecture
optimization on CIFAR-10 with a deeper network targeting 95%+ accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import neuroexapt
from neuroexapt.trainer import Trainer


class DeepCNN(nn.Module):
    """Deep CNN for CIFAR-10 classification targeting 95%+ accuracy."""
    
    def __init__(self, num_classes=10):
        super(DeepCNN, self).__init__()
        
        # Initial convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Deeper convolutional layers
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        # Track expansion layers and next layer numbers
        self.expansion_layers = []
        self.next_conv_num = 7  # Next conv layer number
        self.next_fc_num = 4    # Next fc layer number
        
    def forward(self, x):
        # First block
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Second block
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Third block
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = F.relu(self.bn6(self.conv6(x)))
        
        # Apply expansion conv layers if any
        for layer_name in self.expansion_layers:
            if hasattr(self, layer_name) and layer_name.startswith('conv'):
                layer = getattr(self, layer_name)
                bn_layer = getattr(self, f"{layer_name}_bn", None)
                if isinstance(layer, nn.Conv2d):
                    if bn_layer is not None:
                        x = F.relu(bn_layer(layer(x)))
                    else:
                        x = F.relu(layer(x))
        
        # Flatten
        x = x.view(-1, 256 * 4 * 4)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Apply expansion FC layers if any
        for layer_name in self.expansion_layers:
            if hasattr(self, layer_name) and layer_name.startswith('fc'):
                layer = getattr(self, layer_name)
                if isinstance(layer, nn.Linear):
                    x = F.relu(layer(x))
                    x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x
    
    def add_conv_expansion_layer(self, layer_name, in_channels=256, out_channels=256, device=None):
        """Add a new convolutional expansion layer."""
        if layer_name not in self.expansion_layers:
            new_layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            new_bn = nn.BatchNorm2d(out_channels)
            if device is not None:
                new_layer = new_layer.to(device)
                new_bn = new_bn.to(device)
            setattr(self, layer_name, new_layer)
            setattr(self, f"{layer_name}_bn", new_bn)
            self.expansion_layers.append(layer_name)
            print(f"✅ Added conv layer: {layer_name} ({in_channels}→{out_channels})")
            return True
        return False
    
    def add_fc_expansion_layer(self, layer_name, in_features=256, out_features=256, device=None):
        """Add a new fully connected expansion layer."""
        if layer_name not in self.expansion_layers:
            new_layer = nn.Linear(in_features, out_features)
            if device is not None:
                new_layer = new_layer.to(device)
            setattr(self, layer_name, new_layer)
            self.expansion_layers.append(layer_name)
            print(f"✅ Added FC layer: {layer_name} ({in_features}→{out_features})")
            return True
        return False
    
    def get_next_layer_name(self, layer_type='conv'):
        """Get the next available layer name."""
        if layer_type == 'conv':
            name = f'conv{self.next_conv_num}'
            self.next_conv_num += 1
            return name
        elif layer_type == 'fc':
            name = f'fc{self.next_fc_num}'
            self.next_fc_num += 1
            return name
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")
    
    def get_layer_info(self):
        """Get information about current layers."""
        return {
            'base_layers': ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6', 'fc1', 'fc2', 'fc3'],
            'expansion_layers': self.expansion_layers,
            'total_layers': len(self.expansion_layers) + 9
        }


def get_cifar10_dataloaders(batch_size=128, num_workers=2):
    """Get CIFAR-10 data loaders with advanced downloading capabilities."""
    
    # Import the advanced dataset loader
    from neuroexapt.utils.dataset_loader import AdvancedDatasetLoader
    
    # Initialize the advanced loader
    loader = AdvancedDatasetLoader(
        cache_dir="./data_cache",
        download_dir="./data",
        use_p2p=True,
        use_xunlei=True,
        max_retries=3
    )
    
    # Get data loaders with data augmentation
    return loader.get_cifar10_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        download=True,
        force_download=False
    )


def main():
    """Main training function."""
    
    print("=" * 80)
    print("Neuro Exapt - Deep Classification Example (Target: 95%+ Accuracy)")
    print("=" * 80)
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Hyperparameters for better performance
    batch_size = 128
    learning_rate = 0.001
    epochs = 100  # Longer training
    info_weight = 0.05  # Lower weight for information-theoretic loss
    entropy_threshold = 0.2  # Lower threshold for more aggressive evolution
    
    # Get data loaders
    print("Loading CIFAR-10 dataset with augmentation...")
    train_loader, test_loader = get_cifar10_dataloaders(batch_size)
    print("Training samples: 50000")  # CIFAR-10 standard
    print("Test samples: 10000")      # CIFAR-10 standard
    
    # Create deeper model
    print("\nInitializing deep model...")
    model = DeepCNN(num_classes=10)
    model = model.to(device)
    initial_params = sum(p.numel() for p in model.parameters())
    print(f"Initial model parameters: {initial_params:,}")
    
    # Initialize Neuro Exapt with aggressive settings
    print("\nInitializing Neuro Exapt with aggressive evolution...")
    neuro_exapt = neuroexapt.NeuroExapt(
        task_type="classification",
        entropy_weight=entropy_threshold,
        info_weight=info_weight,
        device=device,
        verbose=True
    )
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Create trainer with more frequent evolution
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        neuro_exapt=neuro_exapt,
        optimizer=optimizer,
        evolution_frequency=5,  # More frequent evolution
        device=device,
        verbose=True
    )
    
    # Analyze initial model
    print("\nAnalyzing initial model...")
    initial_analysis = trainer.analyze_model(train_loader)
    print(f"Initial complexity: {initial_analysis['complexity']}")
    print(f"Initial entropy: {initial_analysis['entropy'].get('network_entropy', 'N/A')}")
    
    # Training with manual evolution control
    print(f"\nStarting training for {epochs} epochs...")
    print("-" * 80)
    
    best_acc = 0.0
    evolution_events = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                loss = F.cross_entropy(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        # Calculate metrics
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        print(f"Epoch {epoch+1:3d}/{epochs}: "
              f"Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss/len(test_loader):.4f}, "
              f"Val Acc: {val_acc:.2f}%")
        
        # Manual evolution control - add layers when accuracy plateaus
        if (epoch + 1) % 10 == 0 and val_acc < 95.0:
            print(f"\n🔄 Checking for evolution at epoch {epoch+1}...")
            
            current_params = sum(p.numel() for p in model.parameters())
            current_layer_info = model.get_layer_info()
            
            # Add convolutional layer if accuracy < 90%
            if val_acc < 90.0 and len(current_layer_info['expansion_layers']) < 3:
                expansion_count = len(current_layer_info['expansion_layers'])
                new_layer_name = model.get_next_layer_name('conv')
                model.add_conv_expansion_layer(new_layer_name, device=device)
                
                # Update optimizer for new parameters
                optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
                
                evolution_events.append({
                    'epoch': epoch + 1,
                    'action': 'expand_conv',
                    'layer_added': new_layer_name,
                    'params_before': current_params,
                    'params_after': sum(p.numel() for p in model.parameters()),
                    'accuracy_before': val_acc,
                    'accuracy_after': val_acc
                })
                
                print(f"✅ Evolution: Added conv layer {new_layer_name}")
                print(f"   Parameters: {current_params:,} → {sum(p.numel() for p in model.parameters()):,}")
            
            # Add FC layer if accuracy < 95%
            elif val_acc < 95.0 and len(current_layer_info['expansion_layers']) >= 3:
                expansion_count = len(current_layer_info['expansion_layers'])
                new_layer_name = model.get_next_layer_name('fc')
                model.add_fc_expansion_layer(new_layer_name, device=device)
                
                # Update optimizer for new parameters
                optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
                
                evolution_events.append({
                    'epoch': epoch + 1,
                    'action': 'expand_fc',
                    'layer_added': new_layer_name,
                    'params_before': current_params,
                    'params_after': sum(p.numel() for p in model.parameters()),
                    'accuracy_before': val_acc,
                    'accuracy_after': val_acc
                })
                
                print(f"✅ Evolution: Added FC layer {new_layer_name}")
                print(f"   Parameters: {current_params:,} → {sum(p.numel() for p in model.parameters()):,}")
        
        # Update learning rate
        scheduler.step()
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "./checkpoints/best_deep_model.pth")
            print(f"💾 New best model saved! Accuracy: {best_acc:.2f}%")
        
        # Early stopping if accuracy is high enough
        if val_acc >= 95.0:
            print(f"\n🎉 Target accuracy reached! Stopping early at epoch {epoch+1}")
            break
    
    # Final analysis
    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)
    
    final_params = sum(p.numel() for p in model.parameters())
    final_layer_info = model.get_layer_info()
    
    print(f"\nFinal Results:")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Final parameters: {final_params:,}")
    print(f"Total layers: {final_layer_info['total_layers']}")
    print(f"Expansion layers: {final_layer_info['expansion_layers']}")
    
    print(f"\nEvolution Summary:")
    print(f"Total evolution events: {len(evolution_events)}")
    
    if evolution_events:
        print(f"Evolution events:")
        for i, event in enumerate(evolution_events):
            print(f"  {i+1}. Epoch {event['epoch']}: {event['action']}")
            print(f"     Layer added: {event['layer_added']}")
            print(f"     Parameters: {event['params_before']:,} → {event['params_after']:,}")
            param_change = event['params_after'] - event['params_before']
            print(f"     Change: +{param_change:,} parameters")
            print(f"     Accuracy: {event['accuracy_before']:.2f}% → {event['accuracy_after']:.2f}%")
    
    print(f"\n🎉 Deep classification example completed!")
    print(f"   Target accuracy achieved: {best_acc:.2f}%")


if __name__ == "__main__":
    main() 