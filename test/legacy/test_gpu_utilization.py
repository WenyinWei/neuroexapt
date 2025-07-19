"""Test script to measure GPU utilization with optimized settings."""

import torch
import torch.nn as nn
import time
import psutil
import GPUtil

# Import our models
from examples.basic_classification import DeeperCNN, MultiBranchCNN

def test_gpu_utilization(model_class, batch_size=512, num_iterations=50):
    """Test GPU utilization with given model and batch size."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Cannot test GPU utilization.")
        return 0, 0, 0  # Return default values instead of None
    
    # Enable optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    
    # Create model
    model = model_class(num_classes=10).to(device)
    model.train()
    
    # Create dummy data
    dummy_input = torch.randn(batch_size, 3, 32, 32, device=device)
    dummy_target = torch.randint(0, 10, (batch_size,), device=device)
    
    # Setup optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    # Enable mixed precision
    scaler = torch.cuda.amp.GradScaler()
    
    # Warmup
    print(f"Warming up {model_class.__name__}...")
    for _ in range(10):
        with torch.cuda.amp.autocast():
            output = model(dummy_input)
            loss = criterion(output, dummy_target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    torch.cuda.synchronize()
    
    # Measure GPU utilization
    print(f"\nMeasuring GPU utilization for {model_class.__name__}...")
    gpu_utils = []
    
    start_time = time.time()
    
    for i in range(num_iterations):
        # Record GPU utilization
        GPUs = GPUtil.getGPUs()
        if GPUs:
            gpu_utils.append(GPUs[0].load * 100)
        
        # Training step
        with torch.cuda.amp.autocast():
            output = model(dummy_input)
            loss = criterion(output, dummy_target)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        if i % 10 == 0:
            print(f"  Iteration {i}/{num_iterations}, Current GPU: {gpu_utils[-1]:.1f}%")
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    # Calculate metrics
    avg_gpu = sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0
    max_gpu = max(gpu_utils) if gpu_utils else 0
    throughput = num_iterations * batch_size / (end_time - start_time)
    
    print(f"\n📊 Results for {model_class.__name__}:")
    print(f"  Average GPU Utilization: {avg_gpu:.1f}%")
    print(f"  Peak GPU Utilization: {max_gpu:.1f}%")
    print(f"  Throughput: {throughput:.1f} samples/second")
    print(f"  Time per iteration: {(end_time - start_time) / num_iterations * 1000:.1f} ms")
    
    return avg_gpu, max_gpu, throughput


def main():
    print("=" * 60)
    print("GPU Utilization Test for NeuroExapt")
    print("=" * 60)
    
    # Test different configurations
    configs = [
        (DeeperCNN, 256),
        (DeeperCNN, 512),
        (DeeperCNN, 1024),
        (MultiBranchCNN, 256),
        (MultiBranchCNN, 512),
    ]
    
    results = []
    
    for model_class, batch_size in configs:
        print(f"\n🧪 Testing {model_class.__name__} with batch_size={batch_size}")
        try:
            avg_gpu, max_gpu, throughput = test_gpu_utilization(model_class, batch_size)
            results.append({
                'model': model_class.__name__,
                'batch_size': batch_size,
                'avg_gpu': avg_gpu,
                'max_gpu': max_gpu,
                'throughput': throughput
            })
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"❌ Out of memory with batch_size={batch_size}")
            else:
                raise
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary of GPU Utilization Tests")
    print("=" * 60)
    
    for result in results:
        print(f"\n{result['model']} (batch_size={result['batch_size']}):")
        print(f"  Average GPU: {result['avg_gpu']:.1f}%")
        print(f"  Peak GPU: {result['max_gpu']:.1f}%")
        print(f"  Throughput: {result['throughput']:.1f} samples/sec")
    
    # Recommendations
    best_config = max(results, key=lambda x: x['avg_gpu'])
    print(f"\n✅ Best configuration for GPU utilization:")
    print(f"   Model: {best_config['model']}")
    print(f"   Batch Size: {best_config['batch_size']}")
    print(f"   GPU Utilization: {best_config['avg_gpu']:.1f}%")


if __name__ == "__main__":
    main() 