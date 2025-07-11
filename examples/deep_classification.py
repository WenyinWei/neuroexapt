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
    """Get CIFAR-10 data loaders with proper device optimization."""
    
    # Use more workers for better GPU utilization (0 on Windows to avoid issues)
    import platform
    if platform.system() == 'Windows':
        # Windows has issues with multiprocessing in DataLoader
        actual_workers = 0
    else:
        # Use more workers on Linux/Mac
        actual_workers = min(num_workers, 4)
    
    # Define transforms
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
    
    # Load datasets
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    # Create data loaders with optimization
    train_loader = DataLoader(
        trainset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=actual_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=actual_workers > 0
    )
    
    test_loader = DataLoader(
        testset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=actual_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=actual_workers > 0
    )
    
    return train_loader, test_loader


def main():
    """Main training function."""
    
    print("=" * 80)
    print("Neuro Exapt - Deep Classification with Intelligent Evolution (Target: 95%+ Accuracy)")
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
    print("\nLoading CIFAR-10 dataset with augmentation...")
    train_loader, test_loader = get_cifar10_dataloaders(batch_size)
    print("Training samples: 50000")  # CIFAR-10 standard
    print("Test samples: 10000")      # CIFAR-10 standard
    
    # Create deeper model
    print("\nInitializing deep model...")
    model = DeepCNN(num_classes=10)
    model = model.to(device)
    initial_params = sum(p.numel() for p in model.parameters())
    print(f"Initial model parameters: {initial_params:,}")
    
    # Initialize Neuro Exapt with intelligent operators
    print("\nInitializing Neuro Exapt with Intelligent Evolution...")
    neuro_exapt = neuroexapt.NeuroExapt(
        task_type="classification",
        entropy_weight=entropy_threshold,
        info_weight=info_weight,
        device=device,
        verbose=True
    )
    
    # Wrap model
    wrapped_model = neuro_exapt.wrap_model(model)
    
    # Check if intelligent operators are available
    if hasattr(neuro_exapt, 'use_intelligent_operators') and neuro_exapt.use_intelligent_operators:
        print("✅ Intelligent operators are enabled!")
        print("   - Smart layer type selection based on information metrics")
        print("   - Adaptive data flow management")
        print("   - Intelligent network expansion")
    else:
        print("⚠️  Intelligent operators not available, using standard evolution")
    
    # Analyze initial architecture
    print("\nAnalyzing initial model characteristics...")
    if hasattr(neuro_exapt, 'analyze_layer_characteristics'):
        layer_chars = neuro_exapt.analyze_layer_characteristics(wrapped_model, train_loader)
        print("\nInitial layer analysis:")
        for i, (name, chars) in enumerate(list(layer_chars.items())[:5]):
            print(f"  {name}:")
            print(f"    Spatial complexity: {chars.get('spatial_complexity', 0):.3f}")
            print(f"    Channel redundancy: {chars.get('channel_redundancy', 0):.3f}")
            print(f"    Information density: {chars.get('information_density', 0):.3f}")
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(wrapped_model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Create trainer with intelligent evolution
    print("\nInitializing trainer with intelligent evolution...")
    trainer = Trainer(
        model=wrapped_model,
        neuro_exapt=neuro_exapt,
        optimizer=optimizer,
        evolution_frequency=5,  # More frequent evolution for deep networks
        device=device,
        verbose=True
    )
    
    # Analyze initial model
    print("\nInitial model analysis...")
    initial_analysis = trainer.analyze_model(train_loader)
    print(f"Initial complexity: {initial_analysis['complexity']}")
    print(f"Initial entropy: {initial_analysis['entropy'].get('network_entropy', 'N/A')}")
    
    # Training with intelligent evolution
    print(f"\n🚀 Starting intelligent training for {epochs} epochs...")
    print("Target: 95%+ accuracy with optimal architecture")
    print("-" * 80)
    
    best_acc = 0.0
    evolution_events = []
    intelligent_decisions = []
    
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
            
            # Calculate loss with information-theoretic components
            ce_loss = F.cross_entropy(outputs, targets)
            info_loss = neuro_exapt.info_theory.compute_information_loss(outputs, targets)
            loss = ce_loss + info_weight * info_loss
            
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
            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = wrapped_model(data)
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
        
        # Intelligent evolution check
        if (epoch + 1) % trainer.evolution_frequency == 0 and val_acc < 95.0:
            print(f"\n🔄 Intelligent evolution check at epoch {epoch+1}...")
            
            current_params = sum(p.numel() for p in wrapped_model.parameters())
            
            # Get current metrics
            current_metrics = neuro_exapt.entropy_ctrl.get_metrics().to_dict()
            performance_metrics = {
                'accuracy': train_acc,
                'val_accuracy': val_acc,
                'loss': train_loss / len(train_loader)
            }
            
            # Re-analyze layer characteristics for intelligent decisions
            if hasattr(neuro_exapt, 'analyze_layer_characteristics'):
                layer_chars = neuro_exapt.analyze_layer_characteristics(wrapped_model, train_loader)
                
                # Add layer characteristics to metrics for intelligent operators
                for name, chars in layer_chars.items():
                    current_metrics[f'{name}_complexity'] = chars.get('spatial_complexity', 0)
                    current_metrics[f'{name}_redundancy'] = chars.get('channel_redundancy', 0)
                    # Add dummy activations for analysis
                    if 'conv' in name:
                        current_metrics[f'{name}_activation'] = torch.randn(1, 256, 4, 4, device=device)
                    else:
                        current_metrics[f'{name}_activation'] = torch.randn(1, 256, device=device)
            
            # Force expansion if accuracy is too low
            force_action = None
            if val_acc < 85.0:
                force_action = 'expand'
                print(f"   Forcing expansion due to low accuracy ({val_acc:.2f}%)")
            elif val_acc < 92.0 and train_acc - val_acc > 5.0:
                force_action = 'expand'
                print(f"   Forcing expansion due to overfitting gap")
            
            # Attempt intelligent evolution
            evolved, evolution_info = neuro_exapt.evolve_structure(
                performance_metrics,
                force_action=force_action
            )
            
            if evolved:
                print(f"✅ Intelligent evolution performed: {evolution_info['action']}")
                
                # Record intelligent decisions
                if 'layer_types_added' in evolution_info:
                    intelligent_decisions.append({
                        'epoch': epoch + 1,
                        'action': evolution_info['action'],
                        'layer_types': evolution_info.get('layer_types_added', {}),
                        'accuracy_before': val_acc,
                        'params_before': current_params
                    })
                    
                    print(f"   Intelligent layer selection:")
                    for layer, layer_type in evolution_info['layer_types_added'].items():
                        print(f"     - {layer}: {layer_type}")
                
                # Update optimizer for new parameters
                optimizer = torch.optim.AdamW(wrapped_model.parameters(), lr=learning_rate, weight_decay=1e-4)
                
                evolution_events.append({
                    'epoch': epoch + 1,
                    'action': evolution_info['action'],
                    'params_before': current_params,
                    'params_after': sum(p.numel() for p in wrapped_model.parameters()),
                    'accuracy_before': val_acc
                })
            else:
                print(f"   No evolution needed (entropy: {current_metrics['current_entropy']:.3f}, threshold: {current_metrics['threshold']:.3f})")
        
        # Update learning rate
        scheduler.step()
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(wrapped_model.state_dict(), "./checkpoints/best_deep_intelligent_model.pth")
            print(f"💾 New best model saved! Accuracy: {best_acc:.2f}%")
        
        # Early stopping if accuracy is high enough
        if val_acc >= 95.0:
            print(f"\n🎉 Target accuracy reached! Stopping at epoch {epoch+1}")
            break
    
    # Final analysis
    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)
    
    final_params = sum(p.numel() for p in wrapped_model.parameters())
    
    print(f"\n📊 Final Results:")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Initial parameters: {initial_params:,}")
    print(f"Final parameters: {final_params:,}")
    print(f"Parameter increase: {final_params - initial_params:,} ({(final_params/initial_params - 1)*100:.1f}%)")
    
    print(f"\n🔄 Evolution Summary:")
    print(f"Total evolution events: {len(evolution_events)}")
    
    if evolution_events:
        print(f"\nEvolution timeline:")
        for i, event in enumerate(evolution_events):
            print(f"  {i+1}. Epoch {event['epoch']}: {event['action']}")
            print(f"     Parameters: {event['params_before']:,} → {event['params_after']:,}")
            param_change = event['params_after'] - event['params_before']
            print(f"     Change: +{param_change:,} parameters")
            print(f"     Accuracy before: {event['accuracy_before']:.2f}%")
    
    if intelligent_decisions:
        print(f"\n🧠 Intelligent Layer Decisions:")
        for i, decision in enumerate(intelligent_decisions):
            print(f"\n  {i+1}. Epoch {decision['epoch']}: {decision['action']}")
            print(f"     Accuracy: {decision['accuracy_before']:.2f}%")
            if decision['layer_types']:
                print(f"     Layer types selected:")
                for layer, layer_type in decision['layer_types'].items():
                    print(f"       - {layer}: {layer_type}")
    
    # Generate visualizations
    print("\n📈 Generating visualizations...")
    os.makedirs("./results", exist_ok=True)
    
    # Plot entropy evolution
    neuro_exapt.entropy_ctrl.plot_history(save_path="./results/deep_intelligent_entropy.png")
    
    print(f"\n🎉 Deep classification with intelligent evolution completed!")
    print(f"   Target accuracy {'achieved' if best_acc >= 95.0 else 'approached'}: {best_acc:.2f}%")
    print(f"   Architecture optimized using intelligent layer selection")


if __name__ == "__main__":
    main() 