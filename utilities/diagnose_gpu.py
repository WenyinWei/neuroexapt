import torch
import os
import sys
import platform
import subprocess

print("=== GPU Diagnosis and Configuration ===\n")

# 1. System Information
print("1. System Information:")
print(f"   Python Version: {sys.version}")
print(f"   PyTorch Version: {torch.__version__}")
print(f"   CUDA Available: {torch.cuda.is_available()}")
print(f"   Platform: {platform.system()} {platform.release()}")
print(f"   OS Environment CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}")

# 2. Force NVIDIA GPU selection on Windows
if platform.system() == "Windows":
    print("\n2. Windows GPU Selection:")
    # Set environment to prefer NVIDIA GPU
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # Explicitly disable Intel GPU
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print(f"   CUDA_DEVICE_ORDER: {os.environ.get('CUDA_DEVICE_ORDER')}")
    print(f"   CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# 3. List all available GPUs
print("\n3. Available GPUs:")
if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f"   Number of GPUs: {device_count}")
    
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        print(f"\n   GPU {i}: {props.name}")
        print(f"      Compute Capability: {props.major}.{props.minor}")
        print(f"      Total Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"      Multi Processor Count: {props.multi_processor_count}")
        print(f"      Clock Rate: {props.clock_rate / 1000:.2f} MHz")
        
        # Check current memory usage
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"      Current Memory Allocated: {allocated:.2f} GB")
        print(f"      Current Memory Reserved: {reserved:.2f} GB")
else:
    print("   No CUDA GPUs available!")

# 4. Test GPU operations
print("\n4. Testing GPU Operations:")

if torch.cuda.is_available():
    # Explicitly select NVIDIA GPU
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    print(f"   Selected Device: {device}")
    print(f"   Current Device: {torch.cuda.current_device()}")
    print(f"   Device Name: {torch.cuda.get_device_name(0)}")
    
    # Test tensor operations
    print("\n   Testing Tensor Operations:")
    try:
        # Create large tensors to actually use GPU
        size = 10000
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        
        print(f"   Created two {size}x{size} tensors on GPU")
        
        # Warm-up
        for _ in range(3):
            _ = torch.matmul(a, b)
        torch.cuda.synchronize()
        
        # Timed operation
        import time
        start = time.time()
        for _ in range(10):
            c = torch.matmul(a, b)
            torch.cuda.synchronize()
        end = time.time()
        
        print(f"   Matrix multiplication time (10 iterations): {end - start:.2f} seconds")
        print(f"   GFLOPS: {(10 * 2 * size**3) / ((end - start) * 1e9):.2f}")
        
        # Check memory after operation
        print(f"\n   Memory after operations:")
        print(f"      Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"      Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        
    except Exception as e:
        print(f"   ERROR during GPU operations: {e}")
else:
    print("   Cannot test - No CUDA GPUs available")

# 5. Check for Intel GPU (Windows)
print("\n5. Checking for Intel GPU interference:")
if platform.system() == "Windows":
    try:
        # Try to get GPU info via Windows
        result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            gpus = [line.strip() for line in result.stdout.split('\n') if line.strip() and line.strip() != 'Name']
            print("   Detected GPUs in system:")
            for gpu in gpus:
                print(f"      - {gpu}")
    except:
        print("   Could not query Windows GPU information")

# 6. PyTorch Backend Information
print("\n6. PyTorch Backend Configuration:")
print(f"   cuDNN Available: {torch.backends.cudnn.is_available()}")
print(f"   cuDNN Version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
print(f"   cuDNN Enabled: {torch.backends.cudnn.enabled}")
print(f"   cuDNN Benchmark: {torch.backends.cudnn.benchmark}")
print(f"   cuDNN Deterministic: {torch.backends.cudnn.deterministic}")

# 7. Environment Variables affecting GPU
print("\n7. Relevant Environment Variables:")
env_vars = ['CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'CUDA_LAUNCH_BLOCKING', 
            'TORCH_CUDA_ARCH_LIST', 'PYTORCH_CUDA_ALLOC_CONF']
for var in env_vars:
    print(f"   {var}: {os.environ.get(var, 'Not Set')}")

print("\n=== Diagnosis Complete ===")

# 8. Recommendations
print("\n8. Recommendations:")
if torch.cuda.is_available():
    if 'Intel' in torch.cuda.get_device_name(0):
        print("   WARNING: Intel GPU detected as CUDA device!")
        print("   - Set CUDA_VISIBLE_DEVICES=0 to force NVIDIA GPU")
        print("   - Restart Python/Jupyter kernel after setting")
    else:
        print("   ✓ NVIDIA GPU correctly detected and functional")
else:
    print("   ✗ No CUDA GPUs detected - Check NVIDIA drivers and CUDA installation") 