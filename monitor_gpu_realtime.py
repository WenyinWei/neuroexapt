"""
Real-time GPU monitoring for Windows
Shows accurate GPU utilization including CUDA Compute usage
"""

import subprocess
import time
import os
import sys
import threading
from datetime import datetime

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil not installed. Install with: pip install psutil")

try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False
    print("Warning: GPUtil not installed. Install with: pip install gputil")

try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False
    print("Warning: pynvml not installed. Install with: pip install nvidia-ml-py")


class GPUMonitor:
    """Real-time GPU monitoring with multiple methods"""
    
    def __init__(self):
        self.monitoring = False
        self.gpu_stats = []
        
    def monitor_with_nvidia_smi(self, interval=1):
        """Monitor using nvidia-smi (most accurate)"""
        print("\n=== GPU Monitoring with nvidia-smi ===")
        print("Press Ctrl+C to stop\n")
        
        # Find nvidia-smi
        nvidia_smi_path = self._find_nvidia_smi()
        if not nvidia_smi_path:
            print("ERROR: nvidia-smi not found!")
            return
            
        try:
            while True:
                # Clear screen (Windows)
                os.system('cls' if os.name == 'nt' else 'clear')
                
                print(f"GPU Status at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("-" * 80)
                
                # Get detailed GPU info
                cmd = [nvidia_smi_path, '--query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit', '--format=csv,noheader,nounits']
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    for line in lines:
                        parts = line.split(', ')
                        if len(parts) >= 9:
                            idx = parts[0]
                            name = parts[1]
                            temp = parts[2]
                            gpu_util = parts[3]
                            mem_util = parts[4]
                            mem_used = float(parts[5])
                            mem_total = float(parts[6])
                            power_draw = parts[7]
                            power_limit = parts[8]
                            
                            print(f"\nGPU {idx}: {name}")
                            print(f"  Temperature: {temp}°C")
                            print(f"  GPU Utilization: {gpu_util}%")
                            print(f"  Memory Utilization: {mem_util}%")
                            print(f"  Memory: {mem_used:.0f}MB / {mem_total:.0f}MB ({mem_used/mem_total*100:.1f}%)")
                            print(f"  Power: {power_draw}W / {power_limit}W")
                
                # Show processes using GPU
                print("\n" + "-" * 80)
                print("GPU Processes:")
                cmd_procs = [nvidia_smi_path, '--query-compute-apps=gpu_name,pid,process_name,used_memory', '--format=csv,noheader,nounits']
                result_procs = subprocess.run(cmd_procs, capture_output=True, text=True)
                
                if result_procs.returncode == 0 and result_procs.stdout.strip():
                    lines = result_procs.stdout.strip().split('\n')
                    for line in lines:
                        parts = line.split(', ')
                        if len(parts) >= 4:
                            gpu_name = parts[0]
                            pid = parts[1]
                            proc_name = parts[2]
                            mem = parts[3]
                            print(f"  [{pid}] {proc_name}: {mem}MB on {gpu_name}")
                else:
                    print("  No active GPU processes")
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")
    
    def monitor_with_pynvml(self, interval=1):
        """Monitor using nvidia-ml-py (pynvml)"""
        if not HAS_PYNVML:
            print("pynvml not available. Install with: pip install nvidia-ml-py")
            return
            
        print("\n=== GPU Monitoring with pynvml ===")
        print("Press Ctrl+C to stop\n")
        
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        
        try:
            while True:
                os.system('cls' if os.name == 'nt' else 'clear')
                print(f"GPU Status at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("-" * 80)
                
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    
                    # Get GPU info
                    name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    
                    # Utilization
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    
                    # Memory
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    
                    # Power
                    try:
                        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                        power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
                    except:
                        power = power_limit = 0
                    
                    print(f"\nGPU {i}: {name}")
                    print(f"  Temperature: {temp}°C")
                    print(f"  GPU Utilization: {util.gpu}%")
                    print(f"  Memory Utilization: {util.memory}%")
                    print(f"  Memory: {mem_info.used/1024**2:.0f}MB / {mem_info.total/1024**2:.0f}MB ({mem_info.used/mem_info.total*100:.1f}%)")
                    if power > 0:
                        print(f"  Power: {power:.1f}W / {power_limit:.1f}W")
                    
                    # Get processes
                    try:
                        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                        if processes:
                            print(f"  Active Processes:")
                            for proc in processes:
                                print(f"    PID {proc.pid}: {proc.usedGpuMemory/1024**2:.0f}MB")
                    except:
                        pass
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")
        finally:
            pynvml.nvmlShutdown()
    
    def monitor_with_gputil(self, interval=1):
        """Monitor using GPUtil"""
        if not HAS_GPUTIL:
            print("GPUtil not available. Install with: pip install gputil")
            return
            
        print("\n=== GPU Monitoring with GPUtil ===")
        print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                os.system('cls' if os.name == 'nt' else 'clear')
                print(f"GPU Status at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("-" * 80)
                
                GPUs = GPUtil.getGPUs()
                for gpu in GPUs:
                    print(f"\nGPU {gpu.id}: {gpu.name}")
                    print(f"  Temperature: {gpu.temperature}°C")
                    print(f"  GPU Load: {gpu.load * 100:.1f}%")
                    print(f"  Memory: {gpu.memoryUsed:.0f}MB / {gpu.memoryTotal:.0f}MB ({gpu.memoryUtil * 100:.1f}%)")
                    
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")
    
    def _find_nvidia_smi(self):
        """Find nvidia-smi executable"""
        # Common locations
        locations = [
            r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
            r"C:\Windows\System32\nvidia-smi.exe",
            "nvidia-smi"  # In PATH
        ]
        
        for loc in locations:
            try:
                result = subprocess.run([loc, '--version'], capture_output=True)
                if result.returncode == 0:
                    return loc
            except:
                continue
        
        # Try to find in NVIDIA driver folder
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\NVIDIA Corporation\Global\FolderPath")
            nv_path = winreg.QueryValueEx(key, "Base")[0]
            winreg.CloseKey(key)
            
            nvsmi_path = os.path.join(nv_path, "NVSMI", "nvidia-smi.exe")
            if os.path.exists(nvsmi_path):
                return nvsmi_path
        except:
            pass
            
        return None
    
    def show_monitoring_comparison(self):
        """Show why Task Manager is inaccurate"""
        print("\n=== Why Windows Task Manager Shows Low GPU Usage ===\n")
        
        print("Windows Task Manager GPU Types:")
        print("1. 3D - Graphics/Gaming workloads")
        print("2. Copy - Memory copy operations")
        print("3. Video Encode - Video encoding")
        print("4. Video Decode - Video decoding")
        print("5. Compute_0/1 - CUDA compute (often missing or inaccurate!)")
        
        print("\nDeep Learning uses CUDA Compute Engine, which Task Manager often:")
        print("- Doesn't show at all")
        print("- Shows incorrectly under '3D'")
        print("- Reports with significant delay")
        
        print("\nRecommended monitoring tools:")
        print("1. nvidia-smi - Most accurate, shows all GPU engines")
        print("2. GPU-Z - Visual tool with detailed sensors")
        print("3. HWiNFO64 - Comprehensive system monitoring")
        print("4. MSI Afterburner - Good for real-time graphs")


def main():
    monitor = GPUMonitor()
    
    print("=== GPU Monitoring Tool ===\n")
    print("Select monitoring method:")
    print("1. nvidia-smi (Most accurate)")
    print("2. pynvml (Python API)")
    print("3. GPUtil (Simple Python library)")
    print("4. Show why Task Manager is inaccurate")
    print("5. Exit")
    
    choice = input("\nEnter choice (1-5): ")
    
    if choice == '1':
        monitor.monitor_with_nvidia_smi()
    elif choice == '2':
        monitor.monitor_with_pynvml()
    elif choice == '3':
        monitor.monitor_with_gputil()
    elif choice == '4':
        monitor.show_monitoring_comparison()
    else:
        print("Exiting...")


if __name__ == "__main__":
    main() 