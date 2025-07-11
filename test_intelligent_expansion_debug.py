"""
Debug intelligent expansion to see why layers aren't being added.
"""

import torch
import torch.nn as nn
from neuroexapt import NeuroExapt
from neuroexapt.core.intelligent_operators import IntelligentExpansionOperator, LayerTypeSelector

# Simple test model
class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return x

def main():
    print("Testing Intelligent Expansion Operator\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TestModel().to(device)
    
    print(f"Initial model:")
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            print(f"  {name}: {module}")
    print(f"Initial parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Create operators
    layer_selector = LayerTypeSelector()
    expansion_op = IntelligentExpansionOperator(
        layer_selector=layer_selector,
        expansion_ratio=0.5,  # Expand 50% of layers
        device=device
    )
    
    # Create metrics with layer importances
    metrics = {
        'layer_importances': {
            'conv1': 0.8,  # High importance
            'conv2': 0.9,  # Very high importance
        },
        'conv1_entropy': 0.5,
        'conv2_entropy': 0.3,
        'conv1_activation': torch.randn(1, 16, 32, 32, device=device),
        'conv2_activation': torch.randn(1, 32, 32, 32, device=device),
        'conv1_complexity': 0.7,
        'conv2_complexity': 0.8,
        'conv1_redundancy': 0.2,
        'conv2_redundancy': 0.6,
        'device': device
    }
    
    # Test expansion
    print("Attempting expansion...")
    expanded_model, info = expansion_op.apply(model, metrics)
    
    print(f"\nExpansion info: {info}")
    
    print(f"\nExpanded model:")
    for name, module in expanded_model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules
            print(f"  {name}: {module}")
    
    print(f"\nFinal parameters: {sum(p.numel() for p in expanded_model.parameters()):,}")
    
    # Check if model actually changed
    if expanded_model is model:
        print("\n⚠️  Model reference unchanged - might need deep copy")
    else:
        print("\n✅ New model created")
    
    # Test layer type selection logic
    print("\n\nTesting layer type selection logic:")
    test_cases = [
        {
            'desc': 'High spatial complexity + high MI',
            'tensor': torch.randn(1, 64, 32, 32),
            'metrics': {'entropy': 0.5, 'mutual_information': 0.8}
        },
        {
            'desc': 'Low entropy',
            'tensor': torch.randn(1, 64, 32, 32),
            'metrics': {'entropy': 0.2, 'mutual_information': 0.5}
        },
        {
            'desc': 'High channel redundancy',
            'tensor': torch.randn(1, 64, 32, 32),
            'metrics': {'entropy': 0.5, 'mutual_information': 0.5}
        }
    ]
    
    for case in test_cases:
        selected = layer_selector.select_layer_type(
            case['tensor'], 
            case['metrics'], 
            'classification'
        )
        print(f"  {case['desc']}: {selected}")

if __name__ == "__main__":
    main() 