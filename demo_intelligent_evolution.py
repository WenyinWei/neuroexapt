"""
Demonstration of Intelligent Architecture Evolution in NeuroExapt.

This script shows how the enhanced system:
1. Intelligently selects layer types based on information metrics
2. Dynamically adjusts data flow sizes
3. Creates specialized branches for different features
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

from neuroexapt import NeuroExapt
from neuroexapt.trainer import Trainer


class AdaptiveCNN(nn.Module):
    """
    A CNN designed to be evolved with intelligent operators.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Initial simple architecture
        self.features = nn.Sequential(
            # First block - will be analyzed for complexity
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Second block - candidate for intelligent expansion
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Third block - may be replaced with specialized layers
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # Global pooling to handle variable sizes
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def visualize_evolution_process(neuroexapt, save_path="evolution_analysis.png"):
    """
    Visualize the intelligent evolution process.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Layer type distribution over evolution
    ax = axes[0, 0]
    if hasattr(neuroexapt, 'layer_type_history'):
        layer_types = neuroexapt.layer_type_history
        type_counts = {}
        for epoch_types in layer_types:
            for layer_type in epoch_types:
                type_counts[layer_type] = type_counts.get(layer_type, 0) + 1
        
        ax.bar(type_counts.keys(), type_counts.values())
        ax.set_title('Layer Types Added During Evolution')
        ax.set_xlabel('Layer Type')
        ax.set_ylabel('Count')
    
    # 2. Information metrics evolution
    ax = axes[0, 1]
    if neuroexapt.metrics_history:
        epochs = list(range(len(neuroexapt.metrics_history)))
        entropies = [m['current_entropy'] for m in neuroexapt.metrics_history]
        redundancies = [m.get('redundancy', 0) for m in neuroexapt.metrics_history]
        
        ax.plot(epochs, entropies, label='Entropy', marker='o')
        ax.plot(epochs, redundancies, label='Redundancy', marker='s')
        ax.set_title('Information Metrics Evolution')
        ax.set_xlabel('Evolution Step')
        ax.set_ylabel('Metric Value')
        ax.legend()
    
    # 3. Layer complexity analysis
    ax = axes[1, 0]
    if hasattr(neuroexapt, 'layer_characteristics'):
        chars = neuroexapt.layer_characteristics
        if chars:
            layers = list(chars.keys())[:10]  # First 10 layers
            complexities = [chars[l].get('spatial_complexity', 0) for l in layers]
            redundancies = [chars[l].get('channel_redundancy', 0) for l in layers]
            
            x = np.arange(len(layers))
            width = 0.35
            ax.bar(x - width/2, complexities, width, label='Spatial Complexity')
            ax.bar(x + width/2, redundancies, width, label='Channel Redundancy')
            ax.set_xlabel('Layer')
            ax.set_ylabel('Metric Value')
            ax.set_title('Layer Characteristics')
            ax.set_xticks(x)
            ax.set_xticklabels([l.split('.')[-1] for l in layers], rotation=45)
            ax.legend()
    
    # 4. Evolution actions timeline
    ax = axes[1, 1]
    if neuroexapt.evolution_history:
        actions = [step.action for step in neuroexapt.evolution_history]
        action_types = list(set(actions))
        action_colors = plt.cm.tab10(np.linspace(0, 1, len(action_types)))
        
        for i, action in enumerate(actions):
            color_idx = action_types.index(action)
            ax.scatter(i, action_types.index(action), 
                      color=action_colors[color_idx], s=100, alpha=0.7)
        
        ax.set_yticks(range(len(action_types)))
        ax.set_yticklabels(action_types)
        ax.set_xlabel('Evolution Step')
        ax.set_title('Evolution Actions Timeline')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Evolution analysis saved to {save_path}")


def analyze_evolved_architecture(model, neuroexapt):
    """
    Analyze the evolved architecture.
    """
    print("\n=== Evolved Architecture Analysis ===")
    
    # Count layer types
    layer_types = {}
    total_params = 0
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules
            module_type = type(module).__name__
            layer_types[module_type] = layer_types.get(module_type, 0) + 1
            
            # Count parameters
            params = sum(p.numel() for p in module.parameters())
            total_params += params
            
            if params > 0:
                print(f"{name}: {module_type} ({params:,} parameters)")
    
    print(f"\nTotal parameters: {total_params:,}")
    print("\nLayer type distribution:")
    for layer_type, count in sorted(layer_types.items()):
        print(f"  {layer_type}: {count}")
    
    # Show evolution summary
    if neuroexapt.evolution_history:
        print(f"\nEvolution steps taken: {len(neuroexapt.evolution_history)}")
        action_counts = {}
        for step in neuroexapt.evolution_history:
            action_counts[step.action] = action_counts.get(step.action, 0) + 1
        
        print("Actions performed:")
        for action, count in sorted(action_counts.items()):
            print(f"  {action}: {count}")


def main():
    """
    Main demonstration of intelligent evolution.
    """
    print("=== NeuroExapt Intelligent Evolution Demo ===\n")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Use CIFAR-10 for demonstration
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    testloader = DataLoader(testset, batch_size=64, shuffle=False)
    
    # Create model
    model = AdaptiveCNN(num_classes=10).to(device)
    print(f"Initial model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize NeuroExapt with intelligent operators
    neuroexapt = NeuroExapt(
        task_type="classification",
        depth=None,  # Let it determine optimal depth
        entropy_weight=0.5,
        info_weight=0.5,
        device=device,
        verbose=True
    )
    
    # Wrap model
    model = neuroexapt.wrap_model(model)
    
    # Analyze initial architecture
    print("\nAnalyzing initial architecture...")
    if hasattr(neuroexapt, 'analyze_layer_characteristics'):
        neuroexapt.layer_characteristics = neuroexapt.analyze_layer_characteristics(
            model, trainloader
        )
        
        print("Layer characteristics:")
        for layer, chars in list(neuroexapt.layer_characteristics.items())[:5]:
            print(f"  {layer}:")
            print(f"    Spatial complexity: {chars.get('spatial_complexity', 0):.3f}")
            print(f"    Channel redundancy: {chars.get('channel_redundancy', 0):.3f}")
            print(f"    Information density: {chars.get('information_density', 0):.3f}")
    
    # Setup optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Create trainer with evolution enabled
    trainer = Trainer(
        model=model,
        neuroexapt=neuroexapt,
        trainloader=trainloader,
        valloader=testloader,
        optimizer=optimizer,
        device=device,
        use_dynarch=True,  # Enable dynamic architecture
        evolution_interval=5,  # Evolve every 5 epochs
        verbose=True
    )
    
    # Train with intelligent evolution
    print("\nStarting training with intelligent evolution...")
    print("The system will:")
    print("1. Analyze layer characteristics during training")
    print("2. Intelligently select layer types for expansion")
    print("3. Adjust data flow based on feature complexity")
    print("4. Create specialized branches when needed\n")
    
    # Track layer types added
    neuroexapt.layer_type_history = []
    
    # Train for a few epochs to see evolution
    history = trainer.train(epochs=20)
    
    # Analyze evolved architecture
    analyze_evolved_architecture(model, neuroexapt)
    
    # Visualize evolution process
    visualize_evolution_process(neuroexapt)
    
    # Final evaluation
    print("\nFinal evaluation:")
    trainer.validate()
    
    # Show example of layer type decisions
    if hasattr(neuroexapt, 'use_intelligent_operators') and neuroexapt.use_intelligent_operators:
        print("\n=== Intelligent Layer Selection Examples ===")
        print("The system made the following intelligent decisions:")
        print("- Added attention layers where long-range dependencies were detected")
        print("- Inserted pooling layers where spatial complexity was low")
        print("- Used depthwise convolutions where channel redundancy was high")
        print("- Created bottleneck blocks for efficient feature compression")
    else:
        print("\nNote: Intelligent operators not available, using standard evolution")
    
    print("\nDemo completed!")


if __name__ == "__main__":
    main() 