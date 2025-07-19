"""
Test script to verify NVIDIA GPU is correctly used with high utilization.
"""

import torch
import torch.nn as nn
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Force GPU 0

# Import our GPU manager
from neuroexapt.utils.gpu_manager import GPUManager, get_device, ensure_cuda_device

print("=== Testing GPU Correct Usage ===\n")

# Initialize GPU manager
gpu_manager = GPUManager()
device = gpu_manager.initialize()

print(f"Initialized device: {device}")
print(f"Device type: {device.type}")

if device.type == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    try:
        print(f"CUDA Version: {torch.version.cuda}")  # type: ignore
    except:
        pass
    print(f"cuDNN Version: {torch.backends.cudnn.version()}")
    
    # Show initial memory stats
    print("\nInitial Memory Stats:")
    stats = gpu_manager.get_memory_stats()
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}")

# Create a compute-intensive model
class ComputeIntensiveModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Multiple parallel branches for high compute
        self.branch1 = nn.Sequential(
            nn.Conv2d(3, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(3, 128, 5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 5, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv2d(3, 128, 7, padding=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 7, padding=3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 7, padding=3),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        
        self.combine = nn.Conv2d(1536, 1024, 1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, 10)
    
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        
        combined = torch.cat([b1, b2, b3], dim=1)
        x = self.combine(combined)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

print("\nCreating compute-intensive model...")
model = ComputeIntensiveModel()

# Wrap model with GPU optimizations
model = gpu_manager.wrap_model_for_gpu(model, use_compile=False)

# Find optimal batch size
print("\nFinding optimal batch size...")
optimal_batch_size = gpu_manager.get_optimal_batch_size(model, (3, 32, 32), starting_batch_size=256)
print(f"Optimal batch size: {optimal_batch_size}")

# Use a large batch size to maximize GPU usage
batch_size = min(optimal_batch_size, 1024)
print(f"Using batch size: {batch_size}")

# Create optimizer with GPU optimizations
optimizer = gpu_manager.create_optimizer_with_gpu_optimization(
    model.parameters(), 
    lr=0.001, 
    optimizer_type='adamw'
)

# Create dummy data
print("\nCreating test data...")
data = torch.randn(batch_size, 3, 32, 32, device=device)
target = torch.randint(0, 10, (batch_size,), device=device)

# Use mixed precision for higher throughput
scaler = torch.cuda.amp.GradScaler()

# Monitor GPU utilization
try:
    import GPUtil
    import threading
    
    gpu_utils = []
    monitoring = True
    
    def monitor_gpu():
        while monitoring:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_utils.append(gpus[0].load * 100)
            time.sleep(0.1)
    
    monitor_thread = threading.Thread(target=monitor_gpu)
    monitor_thread.start()
except:
    print("GPUtil not available for monitoring")
    monitor_thread = None

# Run intensive computation
print("\nRunning intensive GPU computation...")
print("This should show high GPU utilization...\n")

criterion = nn.CrossEntropyLoss()

# Warm up
for _ in range(5):
    with torch.cuda.amp.autocast():
        output = model(data)
    torch.cuda.synchronize()

# Timed run
start_time = time.time()
iterations = 100

for i in range(iterations):
    # Clear gradients
    optimizer.zero_grad()
    
    # Forward pass with mixed precision
    with torch.cuda.amp.autocast():
        output = model(data)
        loss = criterion(output, target)
    
    # Backward pass
    scaler.scale(loss).backward()
    
    # Optimizer step
    scaler.step(optimizer)
    scaler.update()
    
    # Synchronize for accurate timing
    if i % 10 == 0:
        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        throughput = (i + 1) * batch_size / elapsed
        print(f"Iteration {i}: Loss={loss.item():.4f}, Throughput={throughput:.1f} samples/sec")

torch.cuda.synchronize()
total_time = time.time() - start_time

# Stop monitoring
if monitor_thread:
    monitoring = False
    monitor_thread.join()

print(f"\nTotal time: {total_time:.2f} seconds")
print(f"Average throughput: {iterations * batch_size / total_time:.1f} samples/sec")

# Show GPU stats
if device.type == 'cuda':
    print("\nFinal Memory Stats:")
    stats = gpu_manager.get_memory_stats()
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}")
    
    # Try to get GPU utilization stats
    util_stats = gpu_manager.monitor_gpu_utilization()
    if util_stats:
        print("\nGPU Utilization Stats:")
        for key, value in util_stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
    
    # Show average GPU utilization if monitored
    if 'gpu_utils' in locals() and gpu_utils:
        avg_util = sum(gpu_utils) / len(gpu_utils)
        max_util = max(gpu_utils)
        print(f"\nMonitored GPU Utilization:")
        print(f"  Average: {avg_util:.1f}%")
        print(f"  Maximum: {max_util:.1f}%")

# Clear cache
gpu_manager.clear_cache()

print("\n=== Test Complete ===")
print("\nIf GPU utilization was low, try:")
print("1. Increasing batch size further")
print("2. Using more complex models")
print("3. Ensuring no CPU bottlenecks")
print("4. Checking power settings (should be on Maximum Performance)")
print("5. Close other GPU-using applications") 