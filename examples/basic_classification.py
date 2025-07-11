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
        num_workers=num_workers,
        download=True,                 # Automatically download if not present
        force_download=True            # Force re-download to ensure clean dataset
    )


def main():
    """Main training function."""
    
    print("=" * 60)
    print("Neuro Exapt - Enhanced Dynamic Architecture Example")
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
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_cifar10_dataloaders(batch_size)
    print("Training samples: 50000")  # CIFAR-10 standard
    print("Test samples: 10000")      # CIFAR-10 standard
    
    # Create model - start with a simpler base model for evolution
    print("\nInitializing base model...")
    model = SimpleCNN(num_classes=10)  # Start with simple model, let DynArch evolve it
    model = model.to(device)
    print(f"Initial model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Initialize Neuro Exapt
    print("\nInitializing Neuro Exapt with Enhanced DynArch...")
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
    
    # Create trainer with enhanced evolution
    print("\nInitializing enhanced trainer...")
    trainer = Trainer(
        model=model,
        neuro_exapt=neuro_exapt,
        optimizer=optimizer,
        # scheduler=scheduler,  # Skip scheduler for now to avoid type issues
        evolution_frequency=5,  # More frequent evolution
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
    print_architecture(model)
    
    # Training with automatic evolution
    print(f"\nStarting enhanced training for {epochs} epochs...")
    print("🤖 DynArch will automatically evolve the architecture during training")
    print("-" * 60)
    
    try:
        training_history = trainer.fit(
            train_loader=train_loader,
            val_loader=test_loader,
            epochs=epochs,
            eval_metric="accuracy",
            early_stopping_patience=15,
            save_best=True,
            save_path="./checkpoints/best_dynarch_model.pth"
        )
        
        # Final analysis
        print("\n" + "=" * 60)
        print("🎉 Enhanced Training Completed!")
        print("=" * 60)
        
        final_analysis = trainer.analyze_model(test_loader)
        model_summary = trainer.get_model_summary()
        
        print("\n📊 Final Results:")
        print(f"Total parameters: {model_summary['total_parameters']:,}")
        print(f"Evolution events: {model_summary['evolution_events']}")
        print(f"Final entropy: {final_analysis['entropy'].get('network_entropy', 'N/A')}")
        
        # Get DynArch statistics
        dynarch_stats = trainer.dynarch.get_stats()
        print(f"\n🤖 DynArch Performance:")
        print(f"Total evolution steps: {dynarch_stats['evolution_steps']}")
        print(f"Pareto front size: {dynarch_stats['pareto_front_size']}")
        print(f"Experience buffer size: {dynarch_stats['experience_buffer_size']}")
        print(f"Action distribution: {dynarch_stats['action_distribution']}")
        
        # Print final architecture
        print("\n🏗️ Final Architecture:")
        print_architecture(trainer.wrapped_model)
        
        # Visualizations
        print("\nGenerating visualizations...")
        os.makedirs("./results", exist_ok=True)
        
        try:
            trainer.visualize_training("./results/dynarch_training_progress.png")
            trainer.visualize_evolution("./results/dynarch_evolution_history.png")
            neuro_exapt.visualize_evolution("./results/neuro_exapt_evolution.png")
            print("Visualizations saved to ./results/")
        except Exception as e:
            print(f"Could not generate visualizations: {e}")
        
        # Print evolution summary
        if trainer.evolution_events:
            print("\n🔄 Evolution Events Summary:")
            for i, event in enumerate(trainer.evolution_events):
                action = event['action']
                reward = event['info'].get('reward', 0)
                improvements = event['info'].get('metrics_improvement', {})
                
                print(f"  {i+1}. Epoch {event['epoch']}: {action} (Reward: {reward:.3f})")
                for metric, change in improvements.items():
                    if change != 0:
                        sign = "+" if change > 0 else ""
                        print(f"     {metric}: {sign}{change:.3f}")
        else:
            print("\nNo structure evolution occurred during training.")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🚀 Enhanced Dynamic Architecture Example completed!")
    print("This demonstrates the power of information-theoretic RL-guided evolution!")


if __name__ == "__main__":
    main() 