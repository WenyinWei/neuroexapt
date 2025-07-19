#!/usr/bin/env python3
"""
快速超时Batch Size优化器

特点：
1. 线程池 + 超时机制，超时立即停止
2. 敏感的早停检测
3. 快速识别最优batch size
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Optional, Dict

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from neuroexapt.core.model import Network

def test_single_batch(batch_size: int, timeout: float = 10.0) -> Optional[Dict]:
    """测试单个batch size，带超时"""
    
    def _core_test():
        model = Network(C=16, num_classes=10, layers=4, potential_layers=2, 
                       use_gradient_optimized=True, use_optimized_ops=True, 
                       use_lazy_ops=True, use_memory_efficient=True,
                       use_compile=True, quiet=True).cuda()
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = optim.SGD(model.parameters(), lr=0.025, momentum=0.9, weight_decay=3e-4)
        
        test_input = torch.randn(batch_size, 3, 32, 32, device='cuda')
        test_target = torch.randint(0, 10, (batch_size,), device='cuda')
        
        # 预热
        model.train()
        output = model(test_input)
        loss = criterion(output, test_target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        
        # 测试3次
        times = []
        for _ in range(3):
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            model.train()
            output = model(test_input)
            loss = criterion(output, test_target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        avg_time = sum(times) / len(times)
        samples_per_sec = batch_size / avg_time
        
        del model
        torch.cuda.empty_cache()
        
        return {
            'batch_size': batch_size,
            'avg_time': avg_time,
            'peak_memory_mb': peak_memory,
            'samples_per_sec': samples_per_sec
        }
    
    print(f"测试 batch_size={batch_size:3d}... ", end="", flush=True)
    
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_core_test)
            try:
                result = future.result(timeout=timeout)
                print(f"✅ {result['avg_time']:.2f}s/batch, {result['samples_per_sec']:4.0f} samples/s, {result['peak_memory_mb']:4.0f}MB")
                return result
            except TimeoutError:
                future.cancel()
                print(f"🕒 超时 (>{timeout:.0f}s, 内存腾挪)")
                torch.cuda.empty_cache()
                return None
    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        return None

def find_optimal_batch_size(quiet: bool = False) -> int:
    """快速找到最优batch size"""
    if not quiet:
        print("🚀 快速Batch Size优化器")
        print("=" * 50)
    
    # 候选batch sizes
    candidates = [16, 32, 48, 64, 80, 96, 128, 160, 192, 256]
    
    results = []
    peak_throughput = 0
    declining_count = 0
    
    for batch_size in candidates:
        result = test_single_batch(batch_size, timeout=10.0)
        
        if result is not None:
            results.append(result)
            
            # 早停检测
            current_throughput = result['samples_per_sec']
            if current_throughput > peak_throughput:
                peak_throughput = current_throughput
                declining_count = 0
            else:
                declining_count += 1
                decline_ratio = (peak_throughput - current_throughput) / peak_throughput
                
                # 性能下降超过10%或连续2次下降就停止
                if decline_ratio > 0.10 or declining_count >= 2:
                    if not quiet:
                        print(f"🛑 早停: 性能下降{decline_ratio*100:.1f}%")
                    break
        else:
            # 超时或失败，立即停止
            if not quiet:
                print(f"🛑 停止: batch_size={batch_size}超时或失败")
            break
        
        time.sleep(0.1)  # 短暂休息
    
    # 选择最优
    if not results:
        if not quiet:
            print("❌ 没有成功结果，使用默认值32")
        return 32
    
    # 选择吞吐量最高的
    best = max(results, key=lambda x: x['samples_per_sec'])
    optimal_batch_size = best['batch_size']
    
    if not quiet:
        print(f"\n📈 测试结果:")
        print(f"{'Batch':<6} {'时间':<8} {'速度':<10} {'内存':<8}")
        print("-" * 40)
        
        for result in results:
            marker = "🏆" if result == best else ""
            print(f"{result['batch_size']:<6} {result['avg_time']:<8.2f} "
                  f"{result['samples_per_sec']:<10.0f} {result['peak_memory_mb']:<8.0f} {marker}")
        
        print(f"\n🏆 最优选择: batch_size={optimal_batch_size}")
    
    return optimal_batch_size

if __name__ == "__main__":
    find_optimal_batch_size(quiet=False) 