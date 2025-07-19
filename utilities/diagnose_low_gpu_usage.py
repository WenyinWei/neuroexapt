"""
Diagnose low GPU usage during neural network training
"""

import torch
import torch.nn as nn
import time
import os
import subprocess
from typing import Dict, List, Tuple
import numpy as np

# Import our GPU manager
from neuroexapt.utils.gpu_manager import GPUManager


class GPUUsageDiagnostics:
    """Diagnose reasons for low GPU usage"""
    
    def __init__(self):
        self.gpu_manager = GPUManager()
        self.device = self.gpu_manager.initialize()
        self.results: Dict[str, any] = {}
        
    def run_all_diagnostics(self):
        """Run all diagnostic tests"""
        print("=== GPU Usage Diagnostics ===\n")
        
        # 1. Check basic GPU info
        self.check_gpu_info()
        
        # 2. Test data transfer bottleneck
        self.test_data_transfer()
        
        # 3. Test CPU bottleneck
        self.test_cpu_bottleneck()
        
        # 4. Test batch size impact
        self.test_batch_size_impact()
        
        # 5. Test model complexity
        self.test_model_complexity()
        
        # 6. Check power and thermal limits
        self.check_power_thermal_limits()
        
        # 7. Test memory bandwidth
        self.test_memory_bandwidth()
        
        # 8. Check for other GPU processes
        self.check_gpu_processes()
        
        # 9. Test with optimal settings
        self.test_optimal_settings()
        
        # Summary and recommendations
        self.print_summary_and_recommendations()
    
    def check_gpu_info(self):
        """Check basic GPU information"""
        print("1. GPU Information:")
        if self.device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            print(f"   GPU: {gpu_name}")
            
            props = torch.cuda.get_device_properties(0)
            print(f"   Compute Capability: {props.major}.{props.minor}")
            print(f"   Total Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"   Multiprocessors: {props.multi_processor_count}")
            
            # Check if it's a laptop GPU
            if any(keyword in gpu_name.lower() for keyword in ['mobile', 'laptop', 'max-q']):
                print("   ⚠️  Laptop GPU detected - may have power limitations")
                self.results['laptop_gpu'] = True
            else:
                self.results['laptop_gpu'] = False
        print()
    
    def test_data_transfer(self):
        """Test if data transfer is the bottleneck"""
        print("2. Testing Data Transfer Speed:")
        
        sizes = [1, 10, 100, 1000]  # MB
        for size_mb in sizes:
            size = size_mb * 1024 * 1024 // 4  # Convert to float32 elements
            
            # CPU to GPU
            cpu_tensor = torch.randn(size)
            
            start = time.time()
            for _ in range(10):
                gpu_tensor = cpu_tensor.cuda()
                torch.cuda.synchronize()
            cpu_to_gpu_time = (time.time() - start) / 10
            cpu_to_gpu_speed = size_mb / cpu_to_gpu_time
            
            # GPU to CPU
            start = time.time()
            for _ in range(10):
                cpu_back = gpu_tensor.cpu()
                torch.cuda.synchronize()
            gpu_to_cpu_time = (time.time() - start) / 10
            gpu_to_cpu_speed = size_mb / gpu_to_cpu_time
            
            print(f"   {size_mb}MB: CPU→GPU: {cpu_to_gpu_speed:.1f} MB/s, GPU→CPU: {gpu_to_cpu_speed:.1f} MB/s")
            
            if size_mb == 100:  # Check 100MB transfer
                if cpu_to_gpu_speed < 1000:  # Less than 1GB/s
                    self.results['slow_transfer'] = True
                    print("   ⚠️  Slow data transfer detected!")
        print()
    
    def test_cpu_bottleneck(self):
        """Test if CPU is the bottleneck"""
        print("3. Testing CPU vs GPU Speed:")
        
        # Simple computation
        size = 10000
        cpu_tensor = torch.randn(size, size)
        gpu_tensor = cpu_tensor.cuda()
        
        # CPU computation
        start = time.time()
        for _ in range(5):
            _ = torch.matmul(cpu_tensor, cpu_tensor)
        cpu_time = time.time() - start
        
        # GPU computation
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(5):
            _ = torch.matmul(gpu_tensor, gpu_tensor)
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        
        speedup = cpu_time / gpu_time
        print(f"   Matrix multiplication speedup: {speedup:.1f}x")
        print(f"   CPU time: {cpu_time:.3f}s, GPU time: {gpu_time:.3f}s")
        
        if speedup < 10:
            print("   ⚠️  Low GPU speedup - possible CPU bottleneck or GPU underutilization")
            self.results['low_speedup'] = True
        print()
    
    def test_batch_size_impact(self):
        """Test how batch size affects GPU utilization"""
        print("4. Testing Batch Size Impact:")
        
        model = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 10)
        ).cuda()
        
        batch_sizes = [1, 4, 16, 64, 256]
        best_throughput = 0
        best_batch = 0
        
        for batch_size in batch_sizes:
            try:
                data = torch.randn(batch_size, 3, 32, 32, device='cuda')
                
                # Warm up
                for _ in range(5):
                    _ = model(data)
                torch.cuda.synchronize()
                
                # Time
                start = time.time()
                iterations = 20
                for _ in range(iterations):
                    _ = model(data)
                torch.cuda.synchronize()
                elapsed = time.time() - start
                
                throughput = (iterations * batch_size) / elapsed
                print(f"   Batch size {batch_size}: {throughput:.1f} samples/sec")
                
                if throughput > best_throughput:
                    best_throughput = throughput
                    best_batch = batch_size
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"   Batch size {batch_size}: OOM")
                    break
        
        self.results['best_batch_size'] = best_batch
        
        # Check if small batch sizes are being used
        if best_batch >= 64:
            print(f"   ✓ Good - optimal batch size is {best_batch}")
        else:
            print(f"   ⚠️  Small optimal batch size ({best_batch}) limits GPU utilization")
        print()
    
    def test_model_complexity(self):
        """Test if model is too simple for GPU"""
        print("5. Testing Model Complexity Impact:")
        
        # Simple model
        simple_model = nn.Sequential(
            nn.Linear(1000, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        ).cuda()
        
        # Complex model
        complex_model = nn.Sequential(
            nn.Linear(1000, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 10)
        ).cuda()
        
        batch_size = 256
        data = torch.randn(batch_size, 1000, device='cuda')
        
        # Test simple model
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            _ = simple_model(data)
        torch.cuda.synchronize()
        simple_time = time.time() - start
        
        # Test complex model
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            _ = complex_model(data)
        torch.cuda.synchronize()
        complex_time = time.time() - start
        
        print(f"   Simple model: {simple_time:.3f}s")
        print(f"   Complex model: {complex_time:.3f}s")
        print(f"   Ratio: {complex_time/simple_time:.1f}x")
        
        if complex_time < simple_time * 5:
            print("   ⚠️  Model might be too simple to fully utilize GPU")
            self.results['simple_model'] = True
        print()
    
    def check_power_thermal_limits(self):
        """Check if GPU is hitting power or thermal limits"""
        print("6. Checking Power and Thermal Status:")
        
        try:
            # Find nvidia-smi
            nvidia_smi = None
            for path in [r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe", "nvidia-smi"]:
                try:
                    subprocess.run([path, "--version"], capture_output=True, check=True)
                    nvidia_smi = path
                    break
                except:
                    continue
            
            if nvidia_smi:
                # Get current status
                result = subprocess.run(
                    [nvidia_smi, "--query-gpu=temperature.gpu,power.draw,power.limit,clocks.current.sm,clocks.max.sm", 
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True
                )
                
                if result.returncode == 0:
                    values = result.stdout.strip().split(", ")
                    if len(values) >= 5:
                        temp = float(values[0])
                        power_draw = float(values[1])
                        power_limit = float(values[2])
                        current_clock = float(values[3])
                        max_clock = float(values[4])
                        
                        print(f"   Temperature: {temp}°C")
                        print(f"   Power: {power_draw:.1f}W / {power_limit:.1f}W ({power_draw/power_limit*100:.1f}%)")
                        print(f"   Clock: {current_clock:.0f}MHz / {max_clock:.0f}MHz ({current_clock/max_clock*100:.1f}%)")
                        
                        if temp > 80:
                            print("   ⚠️  High temperature - may be thermal throttling")
                            self.results['thermal_throttle'] = True
                        
                        if current_clock < max_clock * 0.9:
                            print("   ⚠️  GPU not running at full clock speed")
                            self.results['low_clock'] = True
        except Exception as e:
            print(f"   Could not check power/thermal status: {e}")
        print()
    
    def test_memory_bandwidth(self):
        """Test GPU memory bandwidth"""
        print("7. Testing GPU Memory Bandwidth:")
        
        # Large tensor to stress memory
        size = 100 * 1024 * 1024 // 4  # 100MB of float32
        a = torch.randn(size, device='cuda')
        b = torch.randn(size, device='cuda')
        
        # Warm up
        for _ in range(10):
            c = a + b
        torch.cuda.synchronize()
        
        # Test
        start = time.time()
        iterations = 100
        for _ in range(iterations):
            c = a + b  # Simple operation, memory bound
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        # Each iteration reads 2 tensors and writes 1 = 300MB
        total_data = iterations * 300  # MB
        bandwidth = total_data / elapsed / 1024  # GB/s
        
        print(f"   Measured bandwidth: {bandwidth:.1f} GB/s")
        
        # Typical bandwidths: GTX 1060: ~192 GB/s, RTX 3090: ~936 GB/s
        if bandwidth < 50:
            print("   ⚠️  Low memory bandwidth detected")
            self.results['low_bandwidth'] = True
        print()
    
    def check_gpu_processes(self):
        """Check if other processes are using the GPU"""
        print("8. Checking for Other GPU Processes:")
        
        try:
            nvidia_smi = None
            for path in [r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe", "nvidia-smi"]:
                try:
                    subprocess.run([path, "--version"], capture_output=True, check=True)
                    nvidia_smi = path
                    break
                except:
                    continue
            
            if nvidia_smi:
                result = subprocess.run(
                    [nvidia_smi, "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True
                )
                
                if result.returncode == 0 and result.stdout.strip():
                    print("   Other processes using GPU:")
                    lines = result.stdout.strip().split('\n')
                    other_processes = False
                    for line in lines:
                        parts = line.split(", ")
                        if len(parts) >= 3:
                            pid, name, mem = parts[0], parts[1], parts[2]
                            print(f"   - PID {pid}: {name} ({mem}MB)")
                            if int(pid) != os.getpid():
                                other_processes = True
                    
                    if other_processes:
                        print("   ⚠️  Other processes are using the GPU")
                        self.results['other_processes'] = True
                else:
                    print("   No other GPU processes detected")
        except Exception as e:
            print(f"   Could not check GPU processes: {e}")
        print()
    
    def test_optimal_settings(self):
        """Test with optimal settings to show potential"""
        print("9. Testing with Optimal Settings:")
        
        # Create a more compute-intensive model
        class OptimalModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    # Multiple parallel paths
                    nn.Conv2d(3, 256, 3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 512, 3, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(512, 1024, 3, padding=1),
                    nn.BatchNorm2d(1024),
                    nn.ReLU(inplace=True),
                )
                self.classifier = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(1024, 512),
                    nn.ReLU(inplace=True),
                    nn.Linear(512, 10)
                )
            
            def forward(self, x):
                x = self.features(x)
                x = self.classifier(x)
                return x
        
        model = OptimalModel().cuda()
        
        # Find optimal batch size
        optimal_batch = self.gpu_manager.get_optimal_batch_size(model, (3, 32, 32), starting_batch_size=128)
        print(f"   Found optimal batch size: {optimal_batch}")
        
        # Test with optimal settings
        data = torch.randn(optimal_batch, 3, 32, 32, device='cuda')
        target = torch.randint(0, 10, (optimal_batch,), device='cuda')
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, fused=True)
        scaler = torch.cuda.amp.GradScaler()
        
        # Warm up
        for _ in range(5):
            with torch.cuda.amp.autocast():
                output = model(data)
                loss = criterion(output, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        torch.cuda.synchronize()
        
        # Time optimal configuration
        start = time.time()
        iterations = 20
        for _ in range(iterations):
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                output = model(data)
                loss = criterion(output, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        throughput = (iterations * optimal_batch) / elapsed
        print(f"   Throughput with optimal settings: {throughput:.1f} samples/sec")
        self.results['optimal_throughput'] = throughput
        print()
    
    def print_summary_and_recommendations(self):
        """Print summary and recommendations"""
        print("\n=== SUMMARY AND RECOMMENDATIONS ===\n")
        
        issues = []
        
        if self.results.get('laptop_gpu'):
            issues.append("• Laptop GPU detected - consider using desktop GPU or cloud computing")
        
        if self.results.get('slow_transfer'):
            issues.append("• Slow CPU-GPU data transfer - use DataLoader with pin_memory=True and num_workers>0")
        
        if self.results.get('low_speedup'):
            issues.append("• Low GPU speedup - ensure operations are GPU-optimized")
        
        if self.results.get('best_batch_size', 256) < 64:
            issues.append("• Small batch size limiting GPU usage - consider gradient accumulation")
        
        if self.results.get('simple_model'):
            issues.append("• Model too simple - GPU works best with larger, more complex models")
        
        if self.results.get('thermal_throttle'):
            issues.append("• GPU thermal throttling - improve cooling or reduce power limit")
        
        if self.results.get('low_clock'):
            issues.append("• GPU not at full clock speed - check power settings and thermal limits")
        
        if self.results.get('low_bandwidth'):
            issues.append("• Low memory bandwidth - possible hardware limitation")
        
        if self.results.get('other_processes'):
            issues.append("• Other processes using GPU - close unnecessary applications")
        
        if issues:
            print("Issues Found:")
            for issue in issues:
                print(issue)
            print()
        else:
            print("No major issues found. GPU should be capable of high utilization.")
            print()
        
        print("General Recommendations:")
        print("1. Use larger batch sizes (use gradient accumulation if OOM)")
        print("2. Use mixed precision training (torch.cuda.amp)")
        print("3. Enable cuDNN benchmark mode: torch.backends.cudnn.benchmark = True")
        print("4. Use DataLoader with pin_memory=True and multiple workers")
        print("5. Ensure Windows Power Plan is set to 'High Performance'")
        print("6. In NVIDIA Control Panel → Manage 3D Settings → Power Management Mode = 'Prefer Maximum Performance'")
        print("7. Monitor with nvidia-smi instead of Task Manager")
        print("8. Consider more complex models or multiple model instances")
        print()
        
        if 'optimal_throughput' in self.results:
            print(f"Your GPU can achieve at least {self.results['optimal_throughput']:.1f} samples/sec with proper configuration!")


def main():
    diagnostics = GPUUsageDiagnostics()
    diagnostics.run_all_diagnostics()


if __name__ == "__main__":
    main() 