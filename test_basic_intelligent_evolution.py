"""
Quick test of intelligent evolution functionality on a small dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from neuroexapt import NeuroExapt
from neuroexapt.trainer import Trainer
from neuroexapt.utils.visualization import print_architecture


class SimpleCNN(nn.Module):
    """Simple CNN for testing."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(32 * 8 * 8, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def create_dummy_data(num_samples=100):
    """Create dummy data for testing."""
    # Random images
    images = torch.randn(num_samples, 3, 32, 32)
    # Random labels
    labels = torch.randint(0, 10, (num_samples,))
    
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=16, shuffle=True)


def main():
    print("=" * 60)
    print("Testing Intelligent Evolution on Small Dataset")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create dummy data
    train_loader = create_dummy_data(100)
    test_loader = create_dummy_data(20)
    
    # Create model
    model = SimpleCNN(num_classes=10).to(device)
    print(f"\nInitial model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize NeuroExapt
    print("\nInitializing NeuroExapt with intelligent operators...")
    neuroexapt = NeuroExapt(
        task_type="classification",
        entropy_weight=0.3,
        info_weight=0.1,
        device=device,
        verbose=True
    )
    
    # Wrap model
    wrapped_model = neuroexapt.wrap_model(model)
    
    # Check intelligent operators
    if hasattr(neuroexapt, 'use_intelligent_operators') and neuroexapt.use_intelligent_operators:
        print("✅ Intelligent operators are enabled!")
    
    # Analyze initial characteristics
    print("\nAnalyzing initial model characteristics...")
    if hasattr(neuroexapt, 'analyze_layer_characteristics'):
        layer_chars = neuroexapt.analyze_layer_characteristics(wrapped_model, train_loader)
        print("\nLayer characteristics:")
        for name, chars in layer_chars.items():
            print(f"  {name}:")
            print(f"    Spatial complexity: {chars.get('spatial_complexity', 0):.3f}")
            print(f"    Channel redundancy: {chars.get('channel_redundancy', 0):.3f}")
            print(f"    Information density: {chars.get('information_density', 0):.3f}")
    
    # Setup optimizer
    optimizer = torch.optim.Adam(wrapped_model.parameters(), lr=0.001)
    
    # Training loop with evolution
    print("\nStarting training with intelligent evolution...")
    
    for epoch in range(10):
        # Training
        wrapped_model.train()
        train_loss = 0.0
        
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = wrapped_model(data)
            
            # Loss with information component
            ce_loss = F.cross_entropy(outputs, targets)
            info_loss = neuroexapt.info_theory.compute_information_loss(outputs, targets)
            loss = ce_loss + 0.1 * info_loss
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        print(f"\nEpoch {epoch+1}/10: Loss = {train_loss/len(train_loader):.4f}")
        
        # Check for evolution every 3 epochs
        if (epoch + 1) % 3 == 0:
            print(f"  Checking for evolution...")
            
            # Get metrics
            current_metrics = neuroexapt.entropy_ctrl.get_metrics().to_dict()
            performance_metrics = {
                'accuracy': 0.5 + epoch * 0.05,  # Dummy increasing accuracy
                'val_accuracy': 0.4 + epoch * 0.05,
                'loss': train_loss / len(train_loader)
            }
            
            # Add layer characteristics for intelligent operators
            if hasattr(neuroexapt, 'analyze_layer_characteristics'):
                layer_chars = neuroexapt.analyze_layer_characteristics(wrapped_model, train_loader)
                for name, chars in layer_chars.items():
                    current_metrics[f'{name}_complexity'] = chars.get('spatial_complexity', 0)
                    current_metrics[f'{name}_redundancy'] = chars.get('channel_redundancy', 0)
                    # Add dummy activations
                    if 'conv' in name:
                        current_metrics[f'{name}_activation'] = torch.randn(1, 32, 16, 16, device=device)
                    else:
                        current_metrics[f'{name}_activation'] = torch.randn(1, 32, device=device)
            
            # Force expansion for demonstration
            evolved, evolution_info = neuroexapt.evolve_structure(
                performance_metrics,
                force_action='expand' if epoch < 6 else None
            )
            
            if evolved:
                print(f"  ✅ Evolution performed: {evolution_info['action']}")
                if 'layer_types_added' in evolution_info:
                    print(f"  Layer types added: {evolution_info['layer_types_added']}")
                
                # Update optimizer
                optimizer = torch.optim.Adam(wrapped_model.parameters(), lr=0.001)
                
                # Show new parameter count
                new_params = sum(p.numel() for p in wrapped_model.parameters())
                print(f"  New parameter count: {new_params:,}")
    
    # Final analysis
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    
    final_params = sum(p.numel() for p in wrapped_model.parameters())
    print(f"\nFinal model parameters: {final_params:,}")
    
    # Show evolution history
    if neuroexapt.evolution_history:
        print(f"\nEvolution events: {len(neuroexapt.evolution_history)}")
        for i, step in enumerate(neuroexapt.evolution_history):
            print(f"  {i+1}. Epoch {step.epoch}: {step.action}")
            print(f"     Parameters: {step.parameters_before:,} → {step.parameters_after:,}")
    
    print("\n✅ Test completed successfully!")


if __name__ == "__main__":
    main() 