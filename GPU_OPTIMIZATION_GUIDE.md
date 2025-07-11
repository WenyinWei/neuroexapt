# GPU Optimization Guide for NeuroExapt

## Current Optimizations Implemented

### 1. Increased Batch Size
- Changed from 128 to 512 for better GPU parallelism
- Consider increasing to 1024 or 2048 if GPU memory allows

### 2. Mixed Precision Training (FP16)
- Automatically enabled when CUDA is available
- Uses torch.cuda.amp for automatic mixed precision
- Typically provides 2-3x speedup on modern GPUs

### 3. Non-blocking Data Transfers
- Uses `non_blocking=True` for CPU-GPU transfers
- Allows computation and data transfer overlap

### 4. Optimized DataLoader
- `pin_memory=True` for faster transfers
- `persistent_workers=True` to avoid worker recreation
- Platform-specific worker count (0 on Windows)

### 5. Larger Model Architecture
- Switched from SimpleCNN to DeeperCNN
- More layers = more computation = better GPU utilization

## Additional Optimizations to Try

### For Even Higher GPU Utilization

```python
# 1. Increase batch size further (if memory allows)
batch_size = 1024  # or even 2048

# 2. Enable cudnn benchmarking
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# 3. Use gradient accumulation for effective larger batches
accumulation_steps = 4
for i, (data, targets) in enumerate(train_loader):
    # ... forward pass ...
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 4. Prefetch next batch while computing current
class DataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_data, self.next_target = next(self.loader)
        except StopIteration:
            self.next_data = None
            self.next_target = None
            return
        
        with torch.cuda.stream(self.stream):
            self.next_data = self.next_data.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        target = self.next_target
        if data is not None:
            data.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return data, target

# 5. Use torch.compile (PyTorch 2.0+)
if hasattr(torch, 'compile'):
    model = torch.compile(model, mode="reduce-overhead")
```

## Monitoring GPU Utilization

```bash
# NVIDIA GPUs
nvidia-smi -l 1  # Update every second

# or use Python
import GPUtil
GPUs = GPUtil.getGPUs()
gpu = GPUs[0]
print(f"GPU Utilization: {gpu.load*100}%")
print(f"GPU Memory: {gpu.memoryUsed}/{gpu.memoryTotal} MB")
```

## Expected GPU Utilization

With all optimizations:
- **SimpleCNN**: 40-60% (too small)
- **DeeperCNN**: 70-85% (good)
- **MultiBranchCNN**: 80-95% (excellent)

## Troubleshooting Low GPU Usage

1. **CPU Bottleneck**: Increase batch size
2. **Data Loading**: Use more workers (Linux/Mac)
3. **Model Too Small**: Use deeper networks
4. **Memory Bound**: Enable mixed precision
5. **Windows Specific**: Consider WSL2 for better performance

## Quick Test Script

```python
import torch
import time

# Test GPU throughput
device = torch.device('cuda')
size = 4096

# Create large tensors
a = torch.randn(size, size, device=device)
b = torch.randn(size, size, device=device)

# Warmup
for _ in range(10):
    c = torch.matmul(a, b)
torch.cuda.synchronize()

# Benchmark
start = time.time()
for _ in range(100):
    c = torch.matmul(a, b)
torch.cuda.synchronize()
end = time.time()

print(f"GPU TFLOPS: {(2 * size**3 * 100) / (end - start) / 1e12:.2f}")
``` 