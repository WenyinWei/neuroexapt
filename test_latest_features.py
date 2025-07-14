"""
Test script to demonstrate NeuroExapt's latest features:
- Colorful architecture visualization with changes
- Batch size optimization with caching
- Entropy-based evolution
- GPU optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuroexapt.neuroexapt import NeuroExapt
from neuroexapt.utils.gpu_manager import gpu_manager
from neuroexapt.utils.visualization import ModelVisualizer
import copy


class TestModel(nn.Module):
    """Simple model for demonstration."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def main():
    print("=" * 80)
    print("🧪 NeuroExapt Latest Features Demonstration")
    print("=" * 80)
    
    # 1. GPU Initialization and Batch Size Optimization
    print("\n📍 Feature 1: GPU Optimization with Batch Size Caching")
    print("-" * 60)
    
    device = gpu_manager.initialize()
    print(f"✅ Device initialized: {device}")
    
    # Create test model
    model = TestModel()
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimize batch size with caching
    print("\n🔍 Finding optimal batch size (with caching)...")
    batch_size = gpu_manager.get_optimal_batch_size(
        model,
        input_shape=(3, 32, 32),
        starting_batch_size=128,
        safety_factor=0.9,
        use_cache=True
    )
    
    # Try again to demonstrate caching
    print("\n🔄 Running again to demonstrate caching...")
    batch_size_2 = gpu_manager.get_optimal_batch_size(
        model,
        input_shape=(3, 32, 32),
        starting_batch_size=128,
        safety_factor=0.9,
        use_cache=True
    )
    
    assert batch_size == batch_size_2, "Cached batch size should match!"
    
    # 2. Colorful Architecture Visualization
    print("\n📍 Feature 2: Colorful Architecture Visualization")
    print("-" * 60)
    
    visualizer = ModelVisualizer()
    
    # Show initial architecture
    print("\n🏗️  Initial Architecture:")
    visualizer.visualize_model(
        model,
        title="Test Model - Initial State",
        sample_input=torch.randn(1, 3, 32, 32)
    )
    
    # Simulate model changes
    original_model = copy.deepcopy(model)
    
    # Modify existing layers (simulating parameter changes during evolution)
    model.fc1 = nn.Linear(32 * 8 * 8, 128)  # Changed from 64 to 128
    model.fc2 = nn.Linear(128, 10)  # Update fc2 to match
    
    # Show architecture with changes highlighted
    print("\n🏗️  Modified Architecture (with changes highlighted):")
    visualizer.visualize_model(
        model,
        previous_model=original_model,
        changed_layers=['fc1', 'fc2'],
        title="Test Model - After Evolution",
        sample_input=torch.randn(1, 3, 32, 32)
    )
    
    # 3. Entropy-Based Evolution
    print("\n📍 Feature 3: Entropy-Based Evolution")
    print("-" * 60)
    
    # Initialize NeuroExapt
    model = model.to(device)
    neuro_exapt = NeuroExapt(
        task_type="classification",
        entropy_weight=0.3,
        info_weight=0.1,
        device=device,
        verbose=True
    )
    
    wrapped_model = neuro_exapt.wrap_model(model)
    
    # Simulate some training steps to generate entropy values
    print("\n📈 Simulating entropy evolution...")
    
    for i in range(5):
        # Simulate forward pass
        dummy_input = torch.randn(4, 3, 32, 32).to(device)
        dummy_target = torch.randint(0, 10, (4,)).to(device)
        
        output = wrapped_model(dummy_input)
        
        # Measure entropy
        current_entropy = neuro_exapt.entropy_ctrl.measure(output)
        threshold = neuro_exapt.entropy_ctrl.threshold
        
        print(f"   Step {i+1}: Entropy = {current_entropy:.3f}, Threshold = {threshold:.3f}")
        
        # Check evolution conditions
        should_prune = neuro_exapt.entropy_ctrl.should_prune(current_entropy)
        should_expand = neuro_exapt.entropy_ctrl.should_expand(current_entropy)
        
        if should_prune:
            print(f"      → Would trigger PRUNING (entropy below threshold)")
        elif should_expand:
            print(f"      → Would trigger EXPANSION (entropy saturated)")
        else:
            print(f"      → No evolution needed")
    
    # 4. Summary of Features
    print("\n📍 Feature Summary")
    print("-" * 60)
    print("✅ GPU optimization with automatic batch size finding")
    print("✅ Batch size caching for faster subsequent runs")
    print("✅ Colorful architecture visualization showing:")
    print("   - 🟢 New layers (green)")
    print("   - 🔴 Removed layers (red with strikethrough)")
    print("   - 🟡 Modified layers (yellow)")
    print("   - 🔵 Unchanged layers (blue)")
    print("✅ Entropy-based evolution decisions (not fixed epochs)")
    print("✅ Information-theoretic optimization")
    
    # 5. Cache Management
    print("\n📍 Cache Management")
    print("-" * 60)
    print(f"📁 Cache directory: {gpu_manager.cache_dir}")
    cache_files = list(gpu_manager.cache_dir.glob("batch_size_*.json"))
    print(f"📦 Cached configurations: {len(cache_files)}")
    
    if cache_files:
        print("\nTo clear cache, run:")
        print("   gpu_manager.clear_batch_size_cache()")
    
    print("\n✅ Demonstration completed!")


if __name__ == "__main__":
    main() 