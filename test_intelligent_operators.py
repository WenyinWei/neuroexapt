"""
Test script to verify intelligent operators functionality.
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from neuroexapt import NeuroExapt
from neuroexapt.core import INTELLIGENT_OPERATORS_AVAILABLE

print("=" * 60)
print("Testing Intelligent Operators")
print("=" * 60)

# Check if intelligent operators are available
print(f"\nIntelligent operators available: {INTELLIGENT_OPERATORS_AVAILABLE}")

if INTELLIGENT_OPERATORS_AVAILABLE:
    from neuroexapt.core.intelligent_operators import (
        LayerTypeSelector,
        IntelligentExpansionOperator,
        AdaptiveDataFlowOperator
    )
    print("✅ Successfully imported intelligent operators!")
else:
    print("❌ Intelligent operators not available")

# Test NeuroExapt initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")

try:
    neuroexapt = NeuroExapt(
        task_type="classification",
        device=device,
        verbose=True
    )
    print("\n✅ NeuroExapt initialized successfully!")
    
    # Check if intelligent operators are enabled
    if hasattr(neuroexapt, 'use_intelligent_operators'):
        print(f"Intelligent operators enabled: {neuroexapt.use_intelligent_operators}")
    
    # Test layer type selection
    if INTELLIGENT_OPERATORS_AVAILABLE:
        print("\nTesting LayerTypeSelector...")
        selector = LayerTypeSelector()
        
        # Create test tensor
        test_tensor = torch.randn(1, 64, 32, 32)
        test_metrics = {
            'entropy': 0.5,
            'mutual_information': 0.7
        }
        
        selected_type = selector.select_layer_type(
            test_tensor,
            test_metrics,
            "classification"
        )
        print(f"Selected layer type: {selected_type}")
        
        # Test with different conditions
        test_cases = [
            # (spatial_complexity, channel_redundancy, entropy, MI) -> expected type
            ("High spatial + high MI", {'entropy': 0.5, 'mutual_information': 0.8}),
            ("Low entropy", {'entropy': 0.2, 'mutual_information': 0.5}),
            ("Low MI + low spatial", {'entropy': 0.5, 'mutual_information': 0.3}),
        ]
        
        print("\nTesting different conditions:")
        for desc, metrics in test_cases:
            selected = selector.select_layer_type(test_tensor, metrics, "classification")
            print(f"  {desc}: {selected}")
    
    # Test a simple model
    print("\n\nTesting with a simple CNN model...")
    
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc = nn.Linear(64 * 8 * 8, 10)
            
        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    model = SimpleCNN().to(device)
    wrapped_model = neuroexapt.wrap_model(model)
    print("✅ Model wrapped successfully!")
    
    # Test analyze_layer_characteristics if available
    if hasattr(neuroexapt, 'analyze_layer_characteristics'):
        print("\nTesting layer characteristic analysis...")
        
        # Create dummy dataloader
        from torch.utils.data import TensorDataset, DataLoader
        dummy_data = torch.randn(10, 3, 32, 32)
        dummy_targets = torch.randint(0, 10, (10,))
        dummy_dataset = TensorDataset(dummy_data, dummy_targets)
        dummy_loader = DataLoader(dummy_dataset, batch_size=2)
        
        layer_chars = neuroexapt.analyze_layer_characteristics(wrapped_model, dummy_loader)
        print(f"Analyzed {len(layer_chars)} layers:")
        for name, chars in list(layer_chars.items())[:3]:
            print(f"  {name}:")
            print(f"    Spatial complexity: {chars.get('spatial_complexity', 0):.3f}")
            print(f"    Channel redundancy: {chars.get('channel_redundancy', 0):.3f}")
    
    print("\n✅ All tests passed!")
    
except Exception as e:
    print(f"\n❌ Error during testing: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Test completed")
print("=" * 60) 