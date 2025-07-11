"""
Basic Classification Example with Neuro Exapt.

This example demonstrates how to use Neuro Exapt for dynamic architecture
optimization on a simple classification task using CIFAR-10.
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
import copy

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import neuroexapt
from neuroexapt.trainer import Trainer
from neuroexapt.utils.visualization import print_architecture


class SimpleCNN(nn.Module):
    """Simple CNN for CIFAR-10 classification."""
    
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling layers
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.25)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
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


class DeeperCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        # Fix: After 5 pooling operations: 32->16->8->4->2->1, so 512*1*1 = 512
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.expansion_layers = []
        self.next_conv_num = 6  # Next conv layer number
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        
        # Apply expansion conv layers if any
        for layer_name in self.expansion_layers:
            if hasattr(self, layer_name) and layer_name.startswith('conv'):
                layer = getattr(self, layer_name)
                if isinstance(layer, nn.Conv2d):
                    x = F.relu(layer(x))
        
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
        
    def add_expansion_layer(self, layer_name, in_channels=512, out_channels=512, device=None):
        if layer_name not in self.expansion_layers:
            new_layer = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            if device is not None:
                new_layer = new_layer.to(device)
            setattr(self, layer_name, new_layer)
            self.expansion_layers.append(layer_name)
            print(f"✅ Added conv layer: {layer_name} ({in_channels}→{out_channels})")
            return True
        return False
        
    def get_next_layer_name(self, layer_type='conv'):
        """Get the next available layer name."""
        if layer_type == 'conv':
            name = f'conv{self.next_conv_num}'
            self.next_conv_num += 1
            return name
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")
        
    def get_layer_info(self):
        return {
            'base_layers': ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc1', 'fc2'],
            'expansion_layers': self.expansion_layers,
            'total_layers': len(self.expansion_layers) + 7
        }


class MultiBranchCNN(nn.Module):
    """Multi-branch CNN with dynamic branching for better feature extraction."""
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Main branch
        self.main_branch = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Secondary branch (for different feature scales)
        self.secondary_branch = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(4, 4),  # Larger pooling for different scale
            
            nn.Conv2d(32, 64, 5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Attention branch (for adaptive feature selection)
        self.attention_branch = nn.Sequential(
            nn.Conv2d(3, 16, 7, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(8, 8),  # Very large pooling for global attention
            
            nn.Conv2d(16, 32, 7, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        )
        
        # Feature fusion layers
        self.fusion_conv = nn.Conv2d(256 + 128 + 32, 512, 1)  # 1x1 conv for fusion
        self.fusion_bn = nn.BatchNorm2d(512)
        
        # Expansion layers (dynamically added)
        self.expansion_layers = []
        self.next_conv_num = 6
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # Main branch
        main_features = self.main_branch(x)
        
        # Secondary branch
        secondary_features = self.secondary_branch(x)
        
        # Attention branch
        attention_features = self.attention_branch(x)
        
        # Upsample attention features to match main branch size
        attention_features = F.interpolate(
            attention_features, 
            size=main_features.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )
        
        # Collect features from all branches
        branch_features = [main_features, secondary_features, attention_features]
        
        # Add dynamically added branches
        for branch_name in ['branch_1', 'branch_2', 'branch_3', 'branch_4', 'branch_5']:
            if hasattr(self, branch_name):
                branch_features.append(getattr(self, branch_name)(x))
        
        # Upsample all branch features to match main branch size
        aligned_features = []
        for features in branch_features:
            if features.shape[2:] != main_features.shape[2:]:
                features = F.interpolate(
                    features, 
                    size=main_features.shape[2:], 
                    mode='bilinear', 
                    align_corners=False
                )
            aligned_features.append(features)
        
        # Concatenate features from all branches
        combined_features = torch.cat(aligned_features, dim=1)
        
        # Fusion
        fused_features = F.relu(self.fusion_bn(self.fusion_conv(combined_features)))
        
        # Apply expansion layers if any
        for layer_name in self.expansion_layers:
            if hasattr(self, layer_name) and layer_name.startswith('conv'):
                layer = getattr(self, layer_name)
                bn_layer = getattr(self, f"{layer_name}_bn", None)
                if isinstance(layer, nn.Conv2d):
                    fused_features = layer(fused_features)
                    if bn_layer is not None:
                        fused_features = bn_layer(fused_features)
                    fused_features = F.relu(fused_features)
        
        # Classification
        output = self.classifier(fused_features)
        return output
        
    def add_expansion_layer(self, layer_name, in_channels=512, out_channels=512, device=None):
        """Add a new convolutional layer to the expansion section."""
        if layer_name not in self.expansion_layers:
            new_layer = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            new_layer_bn = nn.BatchNorm2d(out_channels)
            
            if device is not None:
                new_layer = new_layer.to(device)
                new_layer_bn = new_layer_bn.to(device)
                
            setattr(self, layer_name, new_layer)
            setattr(self, f"{layer_name}_bn", new_layer_bn)
            self.expansion_layers.append(layer_name)
            
            print(f"✅ Added conv layer: {layer_name} ({in_channels}→{out_channels})")
            return True
        return False
        
    def add_branch(self, branch_name, branch_layers, device=None):
        """Add a new branch to the network."""
        if not hasattr(self, branch_name):
            branch = nn.Sequential(*branch_layers)
            if device is not None:
                branch = branch.to(device)
            setattr(self, branch_name, branch)
            
            # Find the output channels of the new branch
            # Look for the last conv layer or adaptive pooling layer
            new_branch_channels = 0
            for layer in reversed(branch_layers):
                if isinstance(layer, nn.Conv2d):
                    new_branch_channels = layer.out_channels
                    break
                elif isinstance(layer, nn.AdaptiveAvgPool2d):
                    # For adaptive pooling, we need to determine the output size
                    # This is a simplified approach - in practice you might need more sophisticated logic
                    new_branch_channels = 96  # Default for our branch design
                    break
            
            # Update fusion layer to accommodate new branch
            current_fusion_input = self.fusion_conv.in_channels
            new_fusion_input = current_fusion_input + new_branch_channels
            
            new_fusion_conv = nn.Conv2d(new_fusion_input, 512, 1)
            new_fusion_bn = nn.BatchNorm2d(512)
            
            if device is not None:
                new_fusion_conv = new_fusion_conv.to(device)
                new_fusion_bn = new_fusion_bn.to(device)
                
            self.fusion_conv = new_fusion_conv
            self.fusion_bn = new_fusion_bn
            
            print(f"✅ Added branch: {branch_name} (channels: {new_branch_channels})")
            return True
        return False
        
    def get_next_layer_name(self, layer_type='conv'):
        """Get the next available layer name."""
        if layer_type == 'conv':
            name = f'conv{self.next_conv_num}'
            self.next_conv_num += 1
            return name
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")
        
    def get_layer_info(self):
        return {
            'base_layers': ['main_branch', 'secondary_branch', 'attention_branch', 'fusion_conv', 'classifier'],
            'expansion_layers': self.expansion_layers,
            'total_layers': len(self.expansion_layers) + 5,
            'branches': ['main', 'secondary', 'attention']
        }
    
    def remove_expansion_layer(self, layer_name):
        if layer_name in self.expansion_layers:
            delattr(self, layer_name)
            if hasattr(self, f"{layer_name}_bn"):
                delattr(self, f"{layer_name}_bn")
            self.expansion_layers.remove(layer_name)
            print(f"❌ Removed expansion layer: {layer_name}")
            return True
        return False
    
    def remove_branch(self, branch_name):
        if hasattr(self, branch_name):
            branch = getattr(self, branch_name)
            # Find output channels
            branch_channels = 0
            for layer in reversed(list(branch)):
                if isinstance(layer, nn.Conv2d):
                    branch_channels = layer.out_channels
                    break
            delattr(self, branch_name)
            # Update fusion
            current_fusion_input = self.fusion_conv.in_channels
            new_fusion_input = current_fusion_input - branch_channels
            self.fusion_conv = nn.Conv2d(new_fusion_input, 512, 1).to(self.fusion_conv.weight.device)
            self.fusion_bn = nn.BatchNorm2d(512).to(self.fusion_bn.weight.device)
            print(f"❌ Removed branch: {branch_name} (channels: {branch_channels})")
            return True
        return False


def get_cifar10_dataloaders(batch_size=128, num_workers=2):
    """Get CIFAR-10 data loaders with advanced downloading capabilities including 迅雷 integration."""
    
    # Import the advanced dataset loader
    from neuroexapt.utils.dataset_loader import AdvancedDatasetLoader
    
    # Use more workers for better GPU utilization (0 on Windows to avoid issues)
    import platform
    if platform.system() == 'Windows':
        # Windows has issues with multiprocessing in DataLoader
        actual_workers = 0
    else:
        # Use more workers on Linux/Mac
        actual_workers = min(num_workers, 4)
    
    # Initialize the advanced loader with P2P acceleration, caching, and 迅雷 integration
    loader = AdvancedDatasetLoader(
        cache_dir="./data_cache",      # Cache directory for downloaded files
        download_dir="./data",         # Directory for extracted datasets
        use_p2p=True,                  # Enable P2P acceleration
        use_xunlei=True,               # Enable 迅雷 integration for Chinese users
        max_retries=3                  # Number of retry attempts
    )
    
    # Get data loaders with automatic downloading and caching
    return loader.get_cifar10_dataloaders(
        batch_size=batch_size,
        num_workers=actual_workers,
        download=True,                 # Automatically download if not present
        force_download=True            # Force re-download to ensure clean dataset
    )


def main():
    """Main training function."""
    
    print("=" * 60)
    print("Neuro Exapt - Intelligent Architecture Evolution Example")
    print("=" * 60)
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Hyperparameters
    batch_size = 128
    learning_rate = 0.001
    epochs = 50
    info_weight = 0.1  # Weight for information-theoretic loss
    entropy_threshold = 0.3  # Threshold for structural decisions
    
    # Get data loaders
    print("\nLoading CIFAR-10 dataset...")
    train_loader, test_loader = get_cifar10_dataloaders(batch_size)
    print("Training samples: 50000")  # CIFAR-10 standard
    print("Test samples: 10000")      # CIFAR-10 standard
    
    # Create model - start with a simpler base model for evolution
    print("\nInitializing base model...")
    model = SimpleCNN(num_classes=10)  # Start with simple model, let intelligent evolution improve it
    model = model.to(device)
    print(f"Initial model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
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
        print("   - LayerTypeSelector will choose optimal layer types")
        print("   - AdaptiveDataFlow will manage feature map sizes")
        print("   - IntelligentExpansion will add appropriate layers")
    else:
        print("⚠️  Intelligent operators not available, using standard evolution")
    
    # Analyze initial architecture characteristics
    print("\nAnalyzing initial model characteristics...")
    if hasattr(neuro_exapt, 'analyze_layer_characteristics'):
        layer_chars = neuro_exapt.analyze_layer_characteristics(wrapped_model, train_loader)
        print("\nInitial layer characteristics:")
        for i, (name, chars) in enumerate(list(layer_chars.items())[:5]):
            print(f"  {name}:")
            print(f"    Spatial complexity: {chars.get('spatial_complexity', 0):.3f}")
            print(f"    Channel redundancy: {chars.get('channel_redundancy', 0):.3f}")
            print(f"    Information density: {chars.get('information_density', 0):.3f}")
            print(f"    Activation sparsity: {chars.get('activation_sparsity', 0):.3f}")
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(wrapped_model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Create trainer with enhanced evolution
    print("\nInitializing enhanced trainer...")
    trainer = Trainer(
        model=wrapped_model,
        neuro_exapt=neuro_exapt,
        optimizer=optimizer,
        evolution_frequency=5,  # More frequent evolution for demonstration
        device=device,
        verbose=True
    )
    
    # Analyze initial model
    print("\nAnalyzing initial model...")
    initial_analysis = trainer.analyze_model(train_loader)
    print(f"Initial complexity: {initial_analysis['complexity']}")
    print(f"Initial entropy: {initial_analysis['entropy'].get('network_entropy', 'N/A')}")
    
    # Print initial architecture
    print("\n🏗️ Initial Architecture:")
    print_architecture(wrapped_model)
    
    # Training with intelligent evolution
    print(f"\nStarting intelligent training for {epochs} epochs...")
    print("🤖 Intelligent evolution will:")
    print("   1. Analyze layer characteristics during training")
    print("   2. Select optimal layer types based on information metrics")
    print("   3. Adjust data flow based on feature complexity")
    print("   4. Add specialized layers where needed")
    print("-" * 60)
    
    # Track evolution decisions
    evolution_decisions = []
    
    try:
        # Manual training loop for better control and debugging
        best_val_acc = 0.0
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
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, targets in test_loader:
                    data, targets = data.to(device), targets.to(device)
                    outputs = wrapped_model(data)
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
            
            train_acc = 100. * train_correct / train_total
            val_acc = 100. * val_correct / val_total
            
            print(f"Epoch {epoch+1:3d}/{epochs}: "
                  f"Train Loss: {train_loss/len(train_loader):.4f}, "
                  f"Train Acc: {train_acc:.2f}%, "
                  f"Val Acc: {val_acc:.2f}%")
            
            # Intelligent evolution check
            if (epoch + 1) % trainer.evolution_frequency == 0:
                print(f"\n🔄 Checking for intelligent evolution at epoch {epoch+1}...")
                
                # Get current metrics
                current_metrics = neuro_exapt.entropy_ctrl.get_metrics().to_dict()
                performance_metrics = {
                    'accuracy': train_acc,
                    'val_accuracy': val_acc,
                    'loss': train_loss / len(train_loader)
                }
                
                # Re-analyze layer characteristics if using intelligent operators
                if hasattr(neuro_exapt, 'analyze_layer_characteristics'):
                    layer_chars = neuro_exapt.analyze_layer_characteristics(wrapped_model, train_loader)
                    
                    # Add layer characteristics to metrics
                    for name, chars in layer_chars.items():
                        current_metrics[f'{name}_complexity'] = chars.get('spatial_complexity', 0)
                        current_metrics[f'{name}_redundancy'] = chars.get('channel_redundancy', 0)
                        current_metrics[f'{name}_activation'] = torch.randn(1, 64, 32, 32, device=device)  # Dummy activation
                
                # Attempt evolution
                evolved, evolution_info = neuro_exapt.evolve_structure(
                    performance_metrics,
                    force_action='expand' if val_acc < 85 else None
                )
                
                if evolved:
                    print(f"✅ Evolution performed: {evolution_info['action']}")
                    if 'layer_types_added' in evolution_info:
                        print(f"   Layer types added: {evolution_info['layer_types_added']}")
                        evolution_decisions.append({
                            'epoch': epoch + 1,
                            'action': evolution_info['action'],
                            'layer_types': evolution_info.get('layer_types_added', {}),
                            'reason': f"Val acc: {val_acc:.2f}%"
                        })
                    
                    # Update optimizer for new parameters
                    optimizer = torch.optim.AdamW(wrapped_model.parameters(), lr=learning_rate, weight_decay=1e-4)
                else:
                    print(f"   No evolution needed at this time")
            
            # Update learning rate
            scheduler.step()
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(wrapped_model.state_dict(), "./checkpoints/best_intelligent_model.pth")
        
        # Final analysis
        print("\n" + "=" * 60)
        print("🎉 Intelligent Training Completed!")
        print("=" * 60)
        
        final_analysis = trainer.analyze_model(test_loader)
        model_summary = trainer.get_model_summary()
        
        print("\n📊 Final Results:")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        print(f"Total parameters: {model_summary['total_parameters']:,}")
        print(f"Evolution events: {model_summary['evolution_events']}")
        print(f"Final entropy: {final_analysis['entropy'].get('network_entropy', 'N/A')}")
        
        # Show intelligent evolution decisions
        if evolution_decisions:
            print("\n🧠 Intelligent Evolution Decisions:")
            for i, decision in enumerate(evolution_decisions):
                print(f"\n  {i+1}. Epoch {decision['epoch']}: {decision['action']}")
                print(f"     Reason: {decision['reason']}")
                if decision['layer_types']:
                    print(f"     Layer types added:")
                    for layer, layer_type in decision['layer_types'].items():
                        print(f"       - {layer}: {layer_type}")
        
        # Print final architecture
        print("\n🏗️ Final Architecture:")
        print_architecture(wrapped_model)
        
        # Visualizations
        print("\nGenerating visualizations...")
        os.makedirs("./results", exist_ok=True)
        
        # Plot entropy history
        neuro_exapt.entropy_ctrl.plot_history(save_path="./results/intelligent_entropy_history.png")
        
        print("\n✅ Example completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 