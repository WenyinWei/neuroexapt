"""
Quick test to demonstrate the improved batch size optimization.
Run with: python test_quick_batch_optimization.py
To skip optimization: SKIP_BATCH_OPTIMIZATION=true python test_quick_batch_optimization.py
"""

import torch
import torch.nn as nn
import time
import os
from neuroexapt.utils.gpu_manager import gpu_manager


class SimpleTestModel(nn.Module):
    """Small model for quick testing"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 10)
        )
        
    def forward(self, x):
        return self.features(x)


def main():
    print("=== Quick Batch Size Optimization Test ===\n")
    
    # Initialize GPU
    device = gpu_manager.initialize()
    print(f"Device: {device}")
    
    if device.type == 'cpu':
        print("\n⚠️  No GPU available, skipping test")
        return
    
    # Get GPU info
    gpu_name = torch.cuda.get_device_name(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {gpu_name}")
    print(f"Total Memory: {total_memory:.1f} GB\n")
    
    # Create model
    model = SimpleTestModel()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    
    # Check if we should skip optimization
    if os.environ.get('SKIP_BATCH_OPTIMIZATION', 'false').lower() == 'true':
        print("\n⚠️  Skipping batch size optimization (SKIP_BATCH_OPTIMIZATION=true)")
        print("Using default batch size: 256")
        return
    
    # Time the optimization
    print("\nStarting batch size optimization...")
    print("-" * 60)
    
    start_time = time.time()
    
    optimal_batch_size = gpu_manager.get_optimal_batch_size(
        model, 
        input_shape=(3, 32, 32),  # CIFAR-10 shape
        starting_batch_size=256,   # Reasonable starting point
        safety_factor=0.9,         # Use 90% of available memory
        max_search_multiplier=4    # Search up to 1024 (256 * 4)
    )
    
    end_time = time.time()
    
    print("-" * 60)
    print(f"\n✅ Optimization completed in {end_time - start_time:.1f} seconds")
    print(f"Recommended batch size: {optimal_batch_size}")
    
    # Show memory stats
    memory_stats = gpu_manager.get_memory_stats()
    if memory_stats:
        print(f"\nCurrent GPU Memory:")
        print(f"  Allocated: {memory_stats['allocated_gb']:.2f} GB")
        print(f"  Reserved:  {memory_stats['reserved_gb']:.2f} GB")
    
    # Test the batch size
    print(f"\nTesting batch size {optimal_batch_size}...")
    try:
        test_input = torch.randn(optimal_batch_size, 3, 32, 32, device=device)
        with torch.no_grad():
            output = model.to(device)(test_input)
        print(f"✅ Successfully processed batch of {optimal_batch_size}")
        print(f"   Output shape: {output.shape}")
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"❌ Batch size {optimal_batch_size} caused OOM (this shouldn't happen!)")
        else:
            raise e
    finally:
        if 'test_input' in locals():
            del test_input
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main() 