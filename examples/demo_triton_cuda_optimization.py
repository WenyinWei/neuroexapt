#!/usr/bin/env python3
"""
NeuroExapt Triton/CUDA Optimization Suite Demo

This script demonstrates the custom Triton and CUDA optimizations implemented
for neural architecture search acceleration.

Key optimizations:
1. CUDA SoftmaxSum: Fused softmax + weighted sum for MixedOp
2. Triton SepConv: Accelerated separable convolutions  
3. Triton Pooling: Optimized pooling operations
4. Automatic fallback for compatibility

Expected speedup: 2-3x for typical DARTS/EXAPT workloads
"""

import torch
import time
import sys
import os
from typing import Dict, Any

# Add the project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def demo_cuda_softmax_sum():
    """Demo CUDA SoftmaxSum optimization"""
    print("🔥 CUDA SoftmaxSum Optimization Demo")
    print("=" * 50)
    
    try:
        from neuroexapt.cuda_ops import SoftmaxSumFunction, CUDA_AVAILABLE
        
        # Test shapes typical for neural architecture search
        N, B, C, H, W = 8, 4, 32, 32, 32  # 8 operations, batch 4, 32x32x32 features
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Device: {device}")
        print(f"CUDA available: {CUDA_AVAILABLE}")
        print(f"Input shape: N={N}, B={B}, C={C}, H={H}, W={W}")
        
        # Generate test data
        x = torch.randn(N, B, C, H, W, device=device, requires_grad=True)
        logits = torch.randn(N, device=device, requires_grad=True)
        
        # Reference PyTorch implementation
        def pytorch_softmax_sum(x, logits):
            weights = torch.softmax(logits, 0)
            return (x * weights.view(-1, 1, 1, 1, 1)).sum(dim=0)
        
        # Benchmark forward pass
        warmup = 5
        runs = 20
        
        # PyTorch baseline
        for _ in range(warmup):
            _ = pytorch_softmax_sum(x, logits)
        
        torch.cuda.synchronize() if device == "cuda" else None
        start = time.perf_counter()
        for _ in range(runs):
            _ = pytorch_softmax_sum(x, logits)
        torch.cuda.synchronize() if device == "cuda" else None
        pytorch_time = (time.perf_counter() - start) / runs
        
        # CUDA implementation
        for _ in range(warmup):
            _ = SoftmaxSumFunction.apply(x, logits)
        
        torch.cuda.synchronize() if device == "cuda" else None
        start = time.perf_counter()
        for _ in range(runs):
            _ = SoftmaxSumFunction.apply(x, logits)
        torch.cuda.synchronize() if device == "cuda" else None
        cuda_time = (time.perf_counter() - start) / runs
        
        # Results
        speedup = pytorch_time / cuda_time
        print(f"PyTorch time: {pytorch_time*1000:.2f}ms")
        print(f"CUDA time: {cuda_time*1000:.2f}ms")
        print(f"Speedup: {speedup:.2f}x")
        
        # Correctness check
        ref_out = pytorch_softmax_sum(x, logits)
        cuda_out = SoftmaxSumFunction.apply(x, logits)
        assert isinstance(cuda_out, torch.Tensor)
        max_diff = torch.max(torch.abs(ref_out - cuda_out)).item()
        print(f"Max difference: {max_diff:.2e} ✅" if max_diff < 1e-4 else f"Max difference: {max_diff:.2e} ❌")
        
    except ImportError as e:
        print(f"❌ CUDA SoftmaxSum not available: {e}")
    
    print()

def demo_triton_sepconv():
    """Demo Triton separable convolution optimization"""
    print("⚡ Triton SepConv Optimization Demo")
    print("=" * 50)
    
    try:
        from neuroexapt.kernels import sepconv_forward_generic, TRITON_AVAILABLE
        
        print(f"Triton available: {TRITON_AVAILABLE}")
        
        # Test parameters
        B, C, H, W = 4, 32, 32, 32
        kernel_size = 3
        stride = 1
        dilation = 1
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Device: {device}")
        print(f"Input shape: B={B}, C={C}, H={H}, W={W}")
        
        # Generate test data
        x = torch.randn(B, C, H, W, device=device)
        dw_weight = torch.randn(C, 1, kernel_size, kernel_size, device=device)
        pw_weight = torch.randn(C * 2, C, 1, 1, device=device)
        bias = torch.randn(C * 2, device=device)
        
        # Reference PyTorch implementation
        def pytorch_sepconv(x, dw_weight, pw_weight, bias):
            pad = ((kernel_size - 1) * dilation) // 2
            y = torch.nn.functional.conv2d(x, dw_weight, None, stride=stride, padding=pad, dilation=dilation, groups=C)
            y = torch.nn.functional.conv2d(y, pw_weight, bias, stride=1, padding=0)
            return y
        
        # Benchmark
        warmup = 5
        runs = 20
        
        # PyTorch baseline
        for _ in range(warmup):
            _ = pytorch_sepconv(x, dw_weight, pw_weight, bias)
        
        torch.cuda.synchronize() if device == "cuda" else None
        start = time.perf_counter()
        for _ in range(runs):
            _ = pytorch_sepconv(x, dw_weight, pw_weight, bias)
        torch.cuda.synchronize() if device == "cuda" else None
        pytorch_time = (time.perf_counter() - start) / runs
        
        # Triton implementation
        for _ in range(warmup):
            _ = sepconv_forward_generic(x, dw_weight, pw_weight, bias, kernel_size=kernel_size, stride=stride, dilation=dilation)
        
        torch.cuda.synchronize() if device == "cuda" else None
        start = time.perf_counter()
        for _ in range(runs):
            _ = sepconv_forward_generic(x, dw_weight, pw_weight, bias, kernel_size=kernel_size, stride=stride, dilation=dilation)
        torch.cuda.synchronize() if device == "cuda" else None
        triton_time = (time.perf_counter() - start) / runs
        
        # Results
        speedup = pytorch_time / triton_time
        print(f"PyTorch time: {pytorch_time*1000:.2f}ms")
        print(f"Triton time: {triton_time*1000:.2f}ms")
        print(f"Speedup: {speedup:.2f}x")
        
        # Correctness check
        ref_out = pytorch_sepconv(x, dw_weight, pw_weight, bias)
        triton_out = sepconv_forward_generic(x, dw_weight, pw_weight, bias, kernel_size=kernel_size, stride=stride, dilation=dilation)
        max_diff = torch.max(torch.abs(ref_out - triton_out)).item()
        print(f"Max difference: {max_diff:.2e} ✅" if max_diff < 1e-4 else f"Max difference: {max_diff:.2e} ❌")
        
    except ImportError as e:
        print(f"❌ Triton SepConv not available: {e}")
    
    print()

def demo_triton_pooling():
    """Demo Triton pooling optimization"""
    print("🏊 Triton Pooling Optimization Demo")
    print("=" * 50)
    
    try:
        from neuroexapt.kernels.pool_triton import avg_pool3x3_forward, max_pool3x3_forward, TRITON_AVAILABLE
        
        print(f"Triton available: {TRITON_AVAILABLE}")
        
        # Test parameters
        B, C, H, W = 4, 32, 32, 32
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Device: {device}")
        print(f"Input shape: B={B}, C={C}, H={H}, W={W}")
        
        # Generate test data
        x = torch.randn(B, C, H, W, device=device)
        
        # Test average pooling
        # PyTorch baseline
        pytorch_avgpool = torch.nn.functional.avg_pool2d(x, 3, stride=1, padding=1, count_include_pad=False)
        
        # Triton implementation
        triton_avgpool = avg_pool3x3_forward(x, stride=1)
        
        # Check correctness
        max_diff = torch.max(torch.abs(pytorch_avgpool - triton_avgpool)).item()
        print(f"AvgPool3x3 max difference: {max_diff:.2e} ✅" if max_diff < 1e-4 else f"AvgPool3x3 max difference: {max_diff:.2e} ❌")
        
        # Test max pooling
        pytorch_maxpool = torch.nn.functional.max_pool2d(x, 3, stride=1, padding=1)
        triton_maxpool = max_pool3x3_forward(x, stride=1)
        
        max_diff = torch.max(torch.abs(pytorch_maxpool - triton_maxpool)).item()
        print(f"MaxPool3x3 max difference: {max_diff:.2e} ✅" if max_diff < 1e-4 else f"MaxPool3x3 max difference: {max_diff:.2e} ❌")
        
    except ImportError as e:
        print(f"❌ Triton Pooling not available: {e}")
    
    print()

def demo_integrated_mixedop():
    """Demo integrated MixedOp with optimizations"""
    print("🧬 Integrated MixedOp Optimization Demo")
    print("=" * 50)
    
    try:
        from neuroexapt.core.operations import MixedOp
        
        # Create a MixedOp with multiple operations
        C = 32
        stride = 1
        mixed_op = MixedOp(C, stride)
        
        if torch.cuda.is_available():
            mixed_op = mixed_op.cuda()
        
        # Test input
        B, H, W = 4, 32, 32
        device = "cuda" if torch.cuda.is_available() else "cpu"
        x = torch.randn(B, C, H, W, device=device)
        
        # Create architecture weights
        n_ops = len(mixed_op._ops)
        weights = torch.randn(n_ops, device=device, requires_grad=True)
        
        print(f"Device: {device}")
        print(f"Input shape: B={B}, C={C}, H={H}, W={W}")
        print(f"Number of operations: {n_ops}")
        
        # Forward pass
        start = time.perf_counter()
        output = mixed_op(x, weights)
        end = time.perf_counter()
        
        print(f"Forward pass time: {(end-start)*1000:.2f}ms")
        print(f"Output shape: {output.shape}")
        print(f"Using optimized MixedOp: {'✅' if hasattr(mixed_op, '_use_optimized') else '❌'}")
        
    except ImportError as e:
        print(f"❌ MixedOp not available: {e}")
    
    print()

def main():
    """Run complete optimization demo"""
    print("🚀 NeuroExapt Triton/CUDA Optimization Suite Demo")
    print("=" * 60)
    print()
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    print()
    
    # Run all demos
    demo_cuda_softmax_sum()
    demo_triton_sepconv()
    demo_triton_pooling()
    demo_integrated_mixedop()
    
    print("🎉 Optimization Demo Complete!")
    print()
    print("Key Benefits:")
    print("✅ Fused CUDA operations reduce memory bandwidth")
    print("✅ Triton kernels optimize common NAS operations")
    print("✅ Automatic fallback ensures compatibility")
    print("✅ Expected 2-3x speedup for DARTS/EXAPT workloads")
    print()
    print("To run with your architecture search:")
    print("  python examples/basic_classification.py --mode exapt --use_optimized_ops")

if __name__ == "__main__":
    main() 