"""
defgroup group_gpu_manager Gpu Manager
ingroup core
Gpu Manager module for NeuroExapt framework.
"""

GPU Manager for NeuroExapt
Ensures proper NVIDIA GPU usage and optimal performance configuration.
"""

import torch
import os
import platform
import warnings
from typing import Optional, Dict, Any, Tuple, Union
import logging
import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


class GPUManager:
    """
    Centralized GPU management for NeuroExapt.
    Ensures NVIDIA GPU is used and provides optimal settings.
    """
    
    def __init__(self):
        self.device: Optional[torch.device] = None
        self.device_properties: Optional[Any] = None  # torch.cuda device properties
        self.is_initialized = False
        self.cache_dir = Path.home() / ".neuroexapt" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def initialize(self, force_gpu_id: Optional[int] = None) -> torch.device:
        """
        Initialize GPU with proper configuration.
        
        Args:
            force_gpu_id: Force specific GPU ID (None for auto-select)
            
        Returns:
            Configured torch device
        """
        # Windows-specific: Force NVIDIA GPU selection
        if platform.system() == "Windows":
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            if 'CUDA_VISIBLE_DEVICES' not in os.environ:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(force_gpu_id or 0)
        
        # Check CUDA availability
        if not torch.cuda.is_available():
            warnings.warn("CUDA not available, falling back to CPU")
            self.device = torch.device('cpu')
            self.is_initialized = True
            return self.device
        
        # Select GPU
        gpu_id = force_gpu_id or 0
        if gpu_id >= torch.cuda.device_count():
            warnings.warn(f"GPU {gpu_id} not available, using GPU 0")
            gpu_id = 0
            
        # Set device
        torch.cuda.set_device(gpu_id)
        self.device = torch.device(f'cuda:{gpu_id}')
        
        # Verify it's NVIDIA GPU
        device_name = torch.cuda.get_device_name(gpu_id)
        if 'Intel' in device_name:
            raise RuntimeError(f"Intel GPU detected: {device_name}. Please configure NVIDIA GPU.")
        
        logger.info(f"Using GPU: {device_name}")
        
        # Store device properties
        self.device_properties = torch.cuda.get_device_properties(gpu_id)
        
        # Configure for optimal performance
        self._configure_cuda_settings()
        
        self.is_initialized = True
        return self.device
    
    def _configure_cuda_settings(self):
        """Configure CUDA for optimal performance."""
        # Enable cuDNN auto-tuner for convolutional networks
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        
        # Set memory allocation strategy
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        
        # Enable TF32 for Ampere GPUs (30xx series and newer)
        if hasattr(torch.backends.cuda, 'matmul'):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = True
            
        # Clear cache to start fresh
        torch.cuda.empty_cache()
        
        logger.info("CUDA settings optimized for performance")
    
    def get_optimal_batch_size(self, model: torch.nn.Module, 
                              input_shape: Tuple[int, ...],
                              starting_batch_size: int = 32,
                              safety_factor: float = 0.9,
                              max_search_multiplier: int = 8,
                              use_cache: bool = True) -> int:
        """
        Find optimal batch size for given model and input shape.
        
        Args:
            model: PyTorch model
            input_shape: Shape of single input (without batch dimension)
            starting_batch_size: Initial batch size to try
            safety_factor: Memory safety factor (0-1)
            max_search_multiplier: Maximum multiplier for search range (default 8x starting size)
            use_cache: Whether to use cached results
            
        Returns:
            Optimal batch size
        """
        if not self.is_initialized:
            self.initialize()
            
        if self.device is None or self.device.type == 'cpu':
            return starting_batch_size
        
        # Check cache if enabled
        if use_cache:
            cached_batch_size = self._check_batch_size_cache(model, input_shape, safety_factor)
            if cached_batch_size is not None:
                return cached_batch_size
            
        model = model.to(self.device)
        model.eval()
        
        # More reasonable search range
        low = max(1, starting_batch_size // 4)
        high = starting_batch_size * max_search_multiplier  # Reduced from 16x
        optimal = starting_batch_size
        
        logger.info("Finding optimal batch size...")
        print(f"Starting batch size search (range: {low} to {high})...")
        
        # Quick initial test with starting batch size
        try:
            torch.cuda.empty_cache()
            dummy_input = torch.randn(starting_batch_size, *input_shape, device=self.device)
            with torch.no_grad():
                _ = model(dummy_input)
            torch.cuda.synchronize()
            
            if self.device_properties is not None:
                memory_reserved = torch.cuda.memory_reserved(self.device)
                total_memory = self.device_properties.total_memory
                memory_usage_ratio = memory_reserved / total_memory
                
                # If starting batch size uses less than 50% memory, increase search
                if memory_usage_ratio < 0.5:
                    low = starting_batch_size
                    print(f"Starting batch size {starting_batch_size} uses only {memory_usage_ratio*100:.1f}% memory, searching higher...")
                # If it uses more than safety factor, decrease search
                elif memory_usage_ratio >= safety_factor:
                    high = starting_batch_size
                    print(f"Starting batch size {starting_batch_size} uses {memory_usage_ratio*100:.1f}% memory, searching lower...")
            
            del dummy_input
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e):
                high = starting_batch_size - 1
                print(f"Starting batch size {starting_batch_size} causes OOM, searching lower...")
            else:
                raise e
        
        # Track iterations for progress
        max_iterations = int(torch.log2(torch.tensor(float(high - low + 1))).ceil().item()) + 1
        max_iterations = min(max_iterations, 10)  # Cap at 10 iterations max
        iteration = 0
        
        # Create progress bar if tqdm is available
        if HAS_TQDM:
            pbar = tqdm(total=max_iterations, desc="Testing batch sizes", unit="iteration")
        
        # Binary search with early stopping
        last_good_memory_usage = 0.0
        
        while low <= high and iteration < max_iterations:
            iteration += 1
            mid = (low + high) // 2
            
            # Print current test
            print(f"\rTesting batch size: {mid}...", end='', flush=True)
            
            try:
                # Clear cache
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Try forward pass with smaller test
                test_size = min(mid, 4)  # Test with smaller batch first
                dummy_input = torch.randn(test_size, *input_shape, device=self.device)
                with torch.no_grad():
                    _ = model(dummy_input)
                del dummy_input
                
                # Now test full size
                dummy_input = torch.randn(mid, *input_shape, device=self.device)
                with torch.no_grad():
                    _ = model(dummy_input)
                torch.cuda.synchronize()
                
                # Check memory usage
                memory_allocated = torch.cuda.memory_allocated(self.device)
                memory_reserved = torch.cuda.memory_reserved(self.device)
                
                if self.device_properties is not None:
                    total_memory = self.device_properties.total_memory
                    memory_usage_ratio = memory_reserved / total_memory
                    
                    # Print memory info
                    memory_pct = memory_usage_ratio * 100
                    print(f"\rBatch size {mid}: Memory usage {memory_pct:.1f}%", end='', flush=True)
                    
                    if memory_usage_ratio < safety_factor:
                        optimal = mid
                        last_good_memory_usage = memory_usage_ratio
                        low = mid + 1
                        print(f" ✓", end='', flush=True)
                        
                        # Early stopping if we're close to target
                        if memory_usage_ratio > safety_factor * 0.85:
                            print(f"\nFound good batch size with {memory_pct:.1f}% memory usage, stopping early.")
                            break
                    else:
                        high = mid - 1
                        print(f" ✗", end='', flush=True)
                else:
                    # If no device properties, accept current and stop
                    optimal = mid
                    print(f"\rBatch size {mid}: ✓", end='', flush=True)
                    break
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    high = mid - 1
                    print(f"\rBatch size {mid}: ✗ OOM", end='', flush=True)
                else:
                    if HAS_TQDM:
                        pbar.close()
                    raise e
            finally:
                # Clean up
                if 'dummy_input' in locals():
                    del dummy_input
                torch.cuda.empty_cache()
            
            # Update progress bar
            if HAS_TQDM:
                pbar.update(1)
        
        # Close progress bar
        if HAS_TQDM:
            pbar.close()
        
        # Clear the line and print final result
        print("\r" + " " * 80 + "\r", end='', flush=True)  # Clear the line
        
        # Memory info for optimal batch size
        if self.device_properties is not None and last_good_memory_usage > 0:
            total_memory = self.device_properties.total_memory
            memory_pct = last_good_memory_usage * 100
            memory_gb = last_good_memory_usage * total_memory / 1024**3
            
            print(f"✅ Optimal batch size found: {optimal}")
            print(f"   Memory usage: {memory_pct:.1f}% ({memory_gb:.1f}GB / {total_memory/1024**3:.1f}GB)")
        else:
            print(f"✅ Optimal batch size found: {optimal}")
        
        logger.info(f"Optimal batch size: {optimal}")
        
        # Cache the result if enabled
        if use_cache:
            self._cache_batch_size(model, input_shape, safety_factor, optimal)
        
        return optimal
    
    def _get_model_hash(self, model: torch.nn.Module, input_shape: Tuple[int, ...]) -> str:
        """Generate a hash for model architecture and input shape."""
        # Create a string representation of model architecture
        model_str = str(model)
        param_count = sum(p.numel() for p in model.parameters())
        
        # Include model structure, parameter count, and input shape
        hash_input = f"{model_str}_{param_count}_{input_shape}"
        
        # Generate hash
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def _get_gpu_hash(self) -> Optional[str]:
        """Generate a hash for current GPU configuration."""
        if self.device is None or self.device.type == 'cpu':
            return None
            
        gpu_name = torch.cuda.get_device_name(self.device)
        total_memory = self.device_properties.total_memory if self.device_properties else 0
        
        gpu_info = f"{gpu_name}_{total_memory}"
        return hashlib.md5(gpu_info.encode()).hexdigest()
    
    def _check_batch_size_cache(self, model: torch.nn.Module, 
                               input_shape: Tuple[int, ...],
                               safety_factor: float) -> Optional[int]:
        """Check if we have a cached batch size for this configuration."""
        model_hash = self._get_model_hash(model, input_shape)
        gpu_hash = self._get_gpu_hash()
        
        if gpu_hash is None:
            return None
        
        cache_file = self.cache_dir / f"batch_size_{model_hash}_{gpu_hash}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                # Check if safety factor matches
                if abs(cache_data['safety_factor'] - safety_factor) < 0.01:
                    batch_size = cache_data['batch_size']
                    gpu_name = torch.cuda.get_device_name(self.device)
                    
                    print(f"\n📦 Using cached optimal batch size: {batch_size}")
                    print(f"   Model: {cache_data['model_params']:,} parameters")
                    print(f"   GPU: {gpu_name}")
                    print(f"   Input shape: {input_shape}")
                    print(f"   Safety factor: {safety_factor * 100:.0f}%")
                    print(f"   ⚠️  If you changed GPU, delete cache at: {self.cache_dir}")
                    print()
                    
                    return batch_size
            except Exception as e:
                logger.warning(f"Failed to load batch size cache: {e}")
        
        return None
    
    def _cache_batch_size(self, model: torch.nn.Module,
                         input_shape: Tuple[int, ...],
                         safety_factor: float,
                         batch_size: int):
        """Cache the optimal batch size for this configuration."""
        model_hash = self._get_model_hash(model, input_shape)
        gpu_hash = self._get_gpu_hash()
        
        if gpu_hash is None:
            return
        
        cache_file = self.cache_dir / f"batch_size_{model_hash}_{gpu_hash}.json"
        
        cache_data = {
            'batch_size': batch_size,
            'safety_factor': safety_factor,
            'input_shape': input_shape,
            'model_params': sum(p.numel() for p in model.parameters()),
            'gpu_name': torch.cuda.get_device_name(self.device),
            'gpu_memory_gb': self.device_properties.total_memory / 1024**3 if self.device_properties else 0,
            'timestamp': str(datetime.now())
        }
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            print(f"💾 Cached optimal batch size for future use")
        except Exception as e:
            logger.warning(f"Failed to cache batch size: {e}")
    
    def clear_batch_size_cache(self):
        """Clear all cached batch sizes."""
        cache_files = list(self.cache_dir.glob("batch_size_*.json"))
        
        for cache_file in cache_files:
            try:
                cache_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete cache file {cache_file}: {e}")
        
        print(f"🗑️  Cleared {len(cache_files)} cached batch size entries")
    
    def create_optimizer_with_gpu_optimization(self, 
                                             parameters,
                                             lr: float = 0.001,
                                             optimizer_type: str = 'adamw') -> torch.optim.Optimizer:
        """
        Create optimizer with GPU-optimized settings.
        
        Args:
            parameters: Model parameters
            lr: Learning rate
            optimizer_type: Type of optimizer
            
        Returns:
            Configured optimizer
        """
        # AdamW with optimized settings for GPU
        if optimizer_type.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                parameters,
                lr=lr,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.01,
                fused=True if self.device is not None and self.device.type == 'cuda' else False  # GPU fusion
            )
        elif optimizer_type.lower() == 'sgd':
            optimizer = torch.optim.SGD(
                parameters,
                lr=lr,
                momentum=0.9,
                weight_decay=0.0001,
                nesterov=True
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
            
        return optimizer
    
    def wrap_model_for_gpu(self, model: torch.nn.Module, 
                          use_compile: bool = True) -> torch.nn.Module:
        """
        Wrap model with GPU optimizations.
        
        Args:
            model: PyTorch model
            use_compile: Whether to use torch.compile (PyTorch 2.0+)
            
        Returns:
            Optimized model
        """
        if not self.is_initialized:
            self.initialize()
            
        if self.device is None:
            self.device = torch.device('cpu')
            
        # Move to GPU
        model = model.to(self.device)
        
        # Use torch.compile for PyTorch 2.0+ (but not on Windows due to Triton issues)
        if use_compile and hasattr(torch, 'compile') and self.device.type == 'cuda' and platform.system() != 'Windows':
            try:
                # torch.compile returns a wrapped model
                compiled_model: torch.nn.Module = torch.compile(model, mode='reduce-overhead')  # type: ignore
                logger.info("Model compiled with torch.compile")
                return compiled_model
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")
        elif use_compile and platform.system() == 'Windows':
            logger.info("Skipping torch.compile on Windows (Triton not supported)")
        
        return model
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current GPU memory statistics."""
        if self.device is None or self.device.type == 'cpu':
            return {}
            
        stats = {
            'allocated_gb': torch.cuda.memory_allocated(self.device) / 1024**3,
            'reserved_gb': torch.cuda.memory_reserved(self.device) / 1024**3,
        }
        
        if self.device_properties is not None:
            stats['total_gb'] = self.device_properties.total_memory / 1024**3
            stats['free_gb'] = (self.device_properties.total_memory - torch.cuda.memory_reserved(self.device)) / 1024**3
            
        return stats
    
    def monitor_gpu_utilization(self) -> Dict[str, Any]:
        """Monitor GPU utilization using nvidia-ml-py."""
        try:
            import pynvml  # type: ignore
            pynvml.nvmlInit()
            
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            # Get utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            # Get memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            # Get temperature
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            
            # Get power
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to Watts
            
            pynvml.nvmlShutdown()
            
            return {
                'gpu_utilization': util.gpu,
                'memory_utilization': util.memory,
                'memory_used_gb': mem_info.used / 1024**3,
                'memory_total_gb': mem_info.total / 1024**3,
                'temperature': temp,
                'power_watts': power
            }
        except Exception as e:
            logger.warning(f"Could not monitor GPU: {e}")
            return {}
    
    def clear_cache(self):
        """Clear GPU cache."""
        if self.device and self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


# Global GPU manager instance
gpu_manager = GPUManager()


def get_device(force_gpu_id: Optional[int] = None) -> torch.device:
    """
    Get properly configured GPU device.
    
    Args:
        force_gpu_id: Force specific GPU ID
        
    Returns:
        Configured device
    """
    if not gpu_manager.is_initialized:
        gpu_manager.initialize(force_gpu_id)
    
    if gpu_manager.device is None:
        # Fallback to CPU if device is still None
        return torch.device('cpu')
        
    return gpu_manager.device


def ensure_cuda_device(tensor_or_module: Union[torch.Tensor, torch.nn.Module]) -> Union[torch.Tensor, torch.nn.Module]:
    """
    Ensure tensor or module is on the correct CUDA device.
    
    Args:
        tensor_or_module: PyTorch tensor or module
        
    Returns:
        Tensor or module on correct device
    """
    device = get_device()
    if hasattr(tensor_or_module, 'to'):
        return tensor_or_module.to(device, non_blocking=True)
    return tensor_or_module 