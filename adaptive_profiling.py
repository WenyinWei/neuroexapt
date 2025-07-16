#!/usr/bin/env python3
"""
自适应架构性能Profiling工具

分析自适应生长架构的各个环节耗时：
1. MixedOp计算开销
2. 架构参数更新开销  
3. SoftmaxSum优化效果
4. 内存使用分析
5. 与固定架构的详细对比
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
import os
from contextlib import contextmanager
from typing import Dict, List, Tuple, Optional
import threading
import psutil
import gc

# Add project to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from neuroexapt.core.model import Network
from neuroexapt.core.architect import Architect
from neuroexapt.core.operations import MixedOp
from neuroexapt.cuda_ops.softmax_sum import SoftmaxSumFunction

class PerformanceProfiler:
    """性能分析器"""
    
    def __init__(self):
        self.timings = {}
        self.memory_stats = {}
        self.call_counts = {}
        
    @contextmanager
    def time_block(self, name: str):
        """测量代码块执行时间"""
        start_time = time.perf_counter()
        start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            elapsed = end_time - start_time
            memory_diff = end_memory - start_memory
            
            if name not in self.timings:
                self.timings[name] = []
                self.memory_stats[name] = []
                self.call_counts[name] = 0
            
            self.timings[name].append(elapsed)
            self.memory_stats[name].append(memory_diff)
            self.call_counts[name] += 1
    
    def get_stats(self) -> Dict:
        """获取统计结果"""
        results = {}
        for name in self.timings:
            times = self.timings[name]
            memories = self.memory_stats[name]
            
            results[name] = {
                'avg_time': sum(times) / len(times),
                'total_time': sum(times),
                'call_count': self.call_counts[name],
                'avg_memory': sum(memories) / len(memories) / 1024 / 1024,  # MB
                'total_memory': sum(memories) / 1024 / 1024,  # MB
            }
        
        return results
    
    def print_report(self):
        """打印性能报告"""
        stats = self.get_stats()
        
        print("\n🔍 性能分析报告")
        print("=" * 80)
        print(f"{'组件':<20} {'调用次数':<8} {'总时间(s)':<12} {'平均时间(ms)':<12} {'内存变化(MB)':<12}")
        print("-" * 80)
        
        # 按总时间排序
        sorted_stats = sorted(stats.items(), key=lambda x: x[1]['total_time'], reverse=True)
        
        for name, stat in sorted_stats:
            print(f"{name:<20} {stat['call_count']:<8} "
                  f"{stat['total_time']:<12.3f} {stat['avg_time']*1000:<12.1f} "
                  f"{stat['avg_memory']:<12.1f}")

def create_test_data(batch_size: int = 96):
    """创建测试数据"""
    return (
        torch.randn(batch_size, 3, 32, 32, device='cuda'),
        torch.randint(0, 10, (batch_size,), device='cuda')
    )

def profile_fixed_architecture(profiler: PerformanceProfiler, epochs: int = 3):
    """分析固定架构性能"""
    print("📊 分析固定架构性能...")
    
    # 创建固定架构（类似basic_classification.py中的FixedNetwork）
    from examples.basic_classification import FixedNetwork
    
    model = FixedNetwork(
        C=16,
        num_classes=10,
        layers=6
    ).cuda()
    
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.025, momentum=0.9, weight_decay=3e-4)
    
    print(f"   固定架构参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试数据
    input_data, target_data = create_test_data()
    
    total_time = 0
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        for batch in range(10):  # 简化测试10个batch
            with profiler.time_block("fixed_forward"):
                logits = model(input_data)
            
            with profiler.time_block("fixed_loss"):
                loss = criterion(logits, target_data)
            
            with profiler.time_block("fixed_backward"):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        epoch_time = time.time() - epoch_start
        total_time += epoch_time
        print(f"   固定架构 Epoch {epoch}: {epoch_time:.2f}s")
    
    avg_epoch_time = total_time / epochs
    print(f"   固定架构平均每epoch: {avg_epoch_time:.2f}s")
    
    return avg_epoch_time

def profile_adaptive_architecture(profiler: PerformanceProfiler, epochs: int = 3):
    """分析自适应架构性能"""
    print("\n📊 分析自适应架构性能...")
    
    # 创建自适应架构
    model = Network(
        C=16,
        num_classes=10,
        layers=6,
        potential_layers=4,
        use_optimized_ops=True,
        use_gradient_optimized=True,
        quiet=True
    ).cuda()
    
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.025, momentum=0.9, weight_decay=3e-4)
    
    # 架构优化器
    class SimpleArgs:
        arch_learning_rate = 3e-4
        arch_weight_decay = 1e-3
        momentum = 0.9
        weight_decay = 3e-4
        learning_rate = 0.025
        
    architect = Architect(model, SimpleArgs())
    architect.criterion = criterion
    
    print(f"   自适应架构参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试数据
    input_data, target_data = create_test_data()
    input_valid, target_valid = create_test_data()
    
    total_time = 0
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        for batch in range(10):  # 简化测试10个batch
            # 架构搜索步骤（每5个batch一次）
            if batch % 5 == 0:
                with profiler.time_block("arch_search_step"):
                    architect.step(
                        input_data, target_data, 
                        input_valid, target_valid,
                        0.025, optimizer, False
                    )
            
            # 权重更新步骤
            with profiler.time_block("adaptive_forward"):
                logits = model(input_data)
            
            with profiler.time_block("adaptive_loss"):
                loss = criterion(logits, target_data)
            
            with profiler.time_block("adaptive_backward"):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        epoch_time = time.time() - epoch_start
        total_time += epoch_time
        print(f"   自适应架构 Epoch {epoch}: {epoch_time:.2f}s")
    
    avg_epoch_time = total_time / epochs
    print(f"   自适应架构平均每epoch: {avg_epoch_time:.2f}s")
    
    return avg_epoch_time

def profile_mixedop_components(profiler: PerformanceProfiler):
    """详细分析MixedOp组件性能"""
    print("\n🔬 详细分析MixedOp组件...")
    
    from neuroexapt.core.operations import OPS
    
    # 测试单个操作
    C = 16
    input_tensor = torch.randn(96, C, 32, 32, device='cuda')
    
    print("   单个操作性能测试:")
    for op_name in ['sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'avg_pool_3x3']:
        op = OPS[op_name](C, 1, False).cuda()
        
        # 预热
        for _ in range(5):
            _ = op(input_tensor)
        
        # 测试
        times = []
        for _ in range(20):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = op(input_tensor)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        
        avg_time = sum(times) / len(times) * 1000
        print(f"     {op_name:<15}: {avg_time:.2f}ms")
    
    # 测试MixedOp整体性能
    print("\n   MixedOp整体性能:")
    
    # 标准MixedOp
    mixedop = MixedOp(C, 1).cuda()
    weights = torch.softmax(torch.randn(len(mixedop._ops), device='cuda'), 0)
    
    with profiler.time_block("mixedop_standard"):
        for _ in range(20):
            _ = mixedop(input_tensor, weights)
    
    # 优化MixedOp
    mixedop_opt = MixedOp(C, 1).cuda()
    
    with profiler.time_block("mixedop_optimized"):
        for _ in range(20):
            _ = mixedop_opt(input_tensor, weights)

def profile_softmax_sum_performance(profiler: PerformanceProfiler):
    """分析SoftmaxSum性能"""
    print("\n🔥 分析SoftmaxSum性能...")
    
    # 不同规模测试
    test_configs = [
        ("Small", 4, 2, 16, 16, 16),
        ("Medium", 8, 4, 32, 32, 32),
        ("Large", 12, 8, 64, 64, 64),
    ]
    
    def pytorch_baseline(x, logits):
        weights = torch.softmax(logits, 0)
        return (x * weights.view(-1, 1, 1, 1, 1)).sum(dim=0)
    
    print("   SoftmaxSum性能对比:")
    for name, N, B, C, H, W in test_configs:
        x = torch.randn(N, B, C, H, W, device='cuda')
        logits = torch.randn(N, device='cuda')
        
        # PyTorch基线
        times_pt = []
        for _ in range(10):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = pytorch_baseline(x, logits)
            torch.cuda.synchronize()
            times_pt.append(time.perf_counter() - start)
        
        # CUDA优化版本
        times_cuda = []
        for _ in range(10):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = SoftmaxSumFunction.apply(x, logits)
            torch.cuda.synchronize()
            times_cuda.append(time.perf_counter() - start)
        
        avg_pt = sum(times_pt) / len(times_pt) * 1000
        avg_cuda = sum(times_cuda) / len(times_cuda) * 1000
        speedup = avg_pt / avg_cuda
        
        print(f"     {name:<8}: PyTorch {avg_pt:.2f}ms, CUDA {avg_cuda:.2f}ms, 加速 {speedup:.2f}x")

def main():
    """主函数"""
    print("🚀 自适应架构性能深度分析")
    print("=" * 80)
    
    profiler = PerformanceProfiler()
    
    # 清理GPU内存
    torch.cuda.empty_cache()
    
    # 1. 分析固定架构
    fixed_time = profile_fixed_architecture(profiler, epochs=2)
    
    # 清理GPU内存
    torch.cuda.empty_cache()
    
    # 2. 分析自适应架构
    adaptive_time = profile_adaptive_architecture(profiler, epochs=2)
    
    # 3. 分析MixedOp组件
    profile_mixedop_components(profiler)
    
    # 4. 分析SoftmaxSum性能
    profile_softmax_sum_performance(profiler)
    
    # 5. 总结报告
    profiler.print_report()
    
    print(f"\n📈 性能对比总结:")
    print("=" * 50)
    print(f"固定架构平均时间:     {fixed_time:.2f}s/epoch")
    print(f"自适应架构平均时间:   {adaptive_time:.2f}s/epoch")
    print(f"性能差距:             {adaptive_time/fixed_time:.1f}x 慢")
    
    # 分析原因
    print(f"\n🔍 性能差距分析:")
    print("=" * 50)
    
    # 参数量分析
    fixed_params = 337866  # 从之前的输出
    adaptive_params = 10196070  # 从之前的输出
    param_ratio = adaptive_params / fixed_params
    
    print(f"参数量差异:           {param_ratio:.1f}x ({adaptive_params:,} vs {fixed_params:,})")
    print(f"计算复杂度:           MixedOp vs 固定操作")
    print(f"架构搜索开销:         额外的梯度计算和参数更新")
    
    print(f"\n💡 优化建议:")
    print("1. 减少MixedOp中的候选操作数量")
    print("2. 使用更高效的架构搜索策略") 
    print("3. 实现动态操作剪枝")
    print("4. 优化SoftmaxSum和Triton内核")

if __name__ == "__main__":
    main() 