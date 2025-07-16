#!/usr/bin/env python3
"""
Performance benchmark for SoftmaxSum CUDA vs PyTorch implementations.
Measures throughput and latency improvements.
"""

import time
import torch
import numpy as np
from typing import Dict, List, Any
import argparse

# Try to import CUDA implementation
try:
    from neuroexapt.cuda_ops import SoftmaxSumFunction, CUDA_AVAILABLE
    BENCHMARK_CUDA = CUDA_AVAILABLE
except ImportError:
    BENCHMARK_CUDA = False
    print("CUDA SoftmaxSum not available, benchmarking PyTorch only")


def pytorch_softmax_sum(x: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Reference PyTorch implementation."""
    weights = torch.softmax(logits, 0)
    return (x * weights.view(-1, 1, 1, 1, 1)).sum(dim=0)


def benchmark_forward(func, x: torch.Tensor, logits: torch.Tensor, warmup: int = 10, runs: int = 100) -> float:
    """Benchmark forward pass only."""
    # Warmup
    for _ in range(warmup):
        _ = func(x, logits)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for _ in range(runs):
        _ = func(x, logits)
    
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    return (end - start) / runs


def benchmark_backward(func, x: torch.Tensor, logits: torch.Tensor, warmup: int = 10, runs: int = 100) -> float:
    """Benchmark forward + backward pass."""
    # Warmup
    for _ in range(warmup):
        x_copy = x.detach().requires_grad_(True)
        logits_copy = logits.detach().requires_grad_(True)
        out = func(x_copy, logits_copy)
        loss = out.sum()
        loss.backward()
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for _ in range(runs):
        x_copy = x.detach().requires_grad_(True)
        logits_copy = logits.detach().requires_grad_(True)
        out = func(x_copy, logits_copy)
        loss = out.sum()
        loss.backward()
    
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    return (end - start) / runs


def run_benchmark_suite(shape_configs: List[tuple], device: str = "cuda") -> Dict[str, Any]:
    """Run complete benchmark suite."""
    results = {}
    
    for config_name, (N, B, C, H, W) in shape_configs:
        print(f"\n📊 Benchmarking {config_name}: N={N}, B={B}, C={C}, H={H}, W={W}")
        
        x = torch.randn(N, B, C, H, W, device=device)
        logits = torch.randn(N, device=device)
        
        config_results: Dict[str, Any] = {"shape": (N, B, C, H, W)}
        
        # PyTorch baseline
        pytorch_fwd = benchmark_forward(pytorch_softmax_sum, x, logits)
        pytorch_bwd = benchmark_backward(pytorch_softmax_sum, x, logits)
        
        config_results["pytorch"] = {
            "forward_ms": pytorch_fwd * 1000,
            "backward_ms": pytorch_bwd * 1000,
        }
        
        print(f"   PyTorch: {pytorch_fwd*1000:.2f}ms fwd, {pytorch_bwd*1000:.2f}ms fwd+bwd")
        
        # CUDA implementation
        if BENCHMARK_CUDA:
            cuda_fwd = benchmark_forward(lambda x, l: SoftmaxSumFunction.apply(x, l), x, logits)
            cuda_bwd = benchmark_backward(lambda x, l: SoftmaxSumFunction.apply(x, l), x, logits)
            
            config_results["cuda"] = {
                "forward_ms": cuda_fwd * 1000,
                "backward_ms": cuda_bwd * 1000,
            }
            
            speedup_fwd = pytorch_fwd / cuda_fwd
            speedup_bwd = pytorch_bwd / cuda_bwd
            
            config_results["speedup"] = {
                "forward": speedup_fwd,
                "backward": speedup_bwd,
            }
            
            print(f"   CUDA:    {cuda_fwd*1000:.2f}ms fwd, {cuda_bwd*1000:.2f}ms fwd+bwd")
            print(f"   Speedup: {speedup_fwd:.2f}x fwd, {speedup_bwd:.2f}x fwd+bwd")
        
        results[config_name] = config_results
    
    return results


def print_summary(results: Dict[str, Any]):
    """Print benchmark summary."""
    print("\n" + "="*80)
    print("📈 BENCHMARK SUMMARY")
    print("="*80)
    
    if BENCHMARK_CUDA:
        avg_speedup_fwd = np.mean([r["speedup"]["forward"] for r in results.values()])
        avg_speedup_bwd = np.mean([r["speedup"]["backward"] for r in results.values()])
        
        print(f"Average CUDA Speedup:")
        print(f"  Forward pass:       {avg_speedup_fwd:.2f}x")
        print(f"  Forward + backward: {avg_speedup_bwd:.2f}x")
        print()
        
        best_config = max(results.keys(), key=lambda k: results[k]["speedup"]["backward"])
        best_speedup = results[best_config]["speedup"]["backward"]
        print(f"Best case ({best_config}): {best_speedup:.2f}x speedup")
        
        # Memory efficiency estimate
        typical_shape = results[list(results.keys())[0]]["shape"]
        N, B, C, H, W = typical_shape
        memory_saved_mb = (N * B * C * H * W * 4) / (1024**2)  # Assume float32
        print(f"Memory bandwidth reduction: ~{memory_saved_mb:.1f}MB per forward pass")
    else:
        print("CUDA implementation not available - install CUDA extension for acceleration")


def main():
    parser = argparse.ArgumentParser(description="Benchmark SoftmaxSum performance")
    parser.add_argument("--device", default="cuda", help="Device to benchmark on")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--runs", type=int, default=100, help="Benchmark iterations")
    args = parser.parse_args()
    
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    # Define benchmark configurations (realistic NAS shapes)
    shape_configs = [
        ("Small NAS", (8, 2, 16, 16, 16)),      # DARTS typical
        ("Medium NAS", (8, 4, 32, 32, 32)),     # Larger search space
        ("Large NAS", (12, 4, 48, 32, 32)),     # Deeper search
        ("Wide NAS", (16, 2, 64, 16, 16)),      # More operations
        ("Deep Feature", (8, 8, 64, 16, 16)),   # Larger batch
    ]
    
    print("🚀 NeuroExapt SoftmaxSum Performance Benchmark")
    print(f"Device: {args.device}")
    print(f"CUDA acceleration: {'✅' if BENCHMARK_CUDA else '❌'}")
    
    results = run_benchmark_suite(shape_configs, args.device)
    print_summary(results)


if __name__ == "__main__":
    main() 