#!/usr/bin/env python3
"""
超时机制的Batch Size优化器

特点：
1. 线程池 + 超时机制避免内存腾挪时长时间等待
2. 敏感的早停检测
3. 快速识别最优batch size
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import gc
import sys
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Optional, Dict, List

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from neuroexapt.core.model import Network

class TimeoutBatchOptimizer:
    """超时机制的batch size优化器"""
    
    def __init__(self, timeout_seconds: float = 12.0):
        self.timeout_seconds = timeout_seconds
        
    def get_gpu_memory_info(self):
        """获取GPU内存信息"""
        if not torch.cuda.is_available():
            return 0, 0, 0
        
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        allocated_mem = torch.cuda.memory_allocated() / 1024 / 1024
        available_mem = total_mem - allocated_mem
        
        return total_mem, available_mem, allocated_mem
    
    def _test_batch_core(self, batch_size: int, model: nn.Module) -> Optional[Dict]:
        """核心测试函数（在线程中运行）"""
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            criterion = nn.CrossEntropyLoss().cuda()
            optimizer = optim.SGD(model.parameters(), lr=0.025, momentum=0.9, weight_decay=3e-4)
            
            # 创建测试数据
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
            
            # 多次测试
            times = []
            for _ in range(3):
                torch.cuda.synchronize()
                start_time = time.perf_counter()
                
                model.train()
                output = model(test_input)
                loss = criterion(output, test_target)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                torch.cuda.synchronize()
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            
            # 统计
            peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            avg_time = sum(times) / len(times)
            time_std = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
            time_variance = time_std / avg_time if avg_time > 0 else 0
            samples_per_sec = batch_size / avg_time
            
            return {
                'batch_size': batch_size,
                'avg_time': avg_time,
                'time_variance': time_variance,
                'peak_memory_mb': peak_memory_mb,
                'samples_per_sec': samples_per_sec
            }
            
        except Exception:
            return None
    
    def test_batch_with_timeout(self, batch_size: int, model: nn.Module) -> Optional[Dict]:
        """带超时的batch测试"""
        print(f"测试 batch_size={batch_size:3d}... ", end="", flush=True)
        
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._test_batch_core, batch_size, model)
                
                try:
                    result = future.result(timeout=self.timeout_seconds)
                    
                    if result is not None:
                        print(f"✅ {result['avg_time']:.2f}s/batch, {result['samples_per_sec']:4.0f} samples/s, {result['peak_memory_mb']:4.0f}MB")
                        return result
                    else:
                        print(f"❌ 测试失败")
                        return None
                        
                                 except TimeoutError:
                     future.cancel()
                     print(f"🕒 超时 (>{self.timeout_seconds:.0f}s, 内存腾挪)")
                     torch.cuda.empty_cache()
                     return None
                    
        except Exception as e:
            print(f"❌ 错误: {str(e)}")
            return None
    
    def find_optimal_batch_size(self, quiet: bool = False) -> int:
        """寻找最优batch size"""
        if not quiet:
            print("🧠 超时机制Batch Size优化器")
            print("=" * 60)
        
        # 创建测试模型
        model = Network(
            C=16,
            num_classes=10,
            layers=4,  # 减少到4层加快测试
            potential_layers=2,
            use_gradient_optimized=True,
            quiet=True
        ).cuda()
        
        # 获取内存信息
        total_mem, available_mem, used_mem = self.get_gpu_memory_info()
        if not quiet:
            print(f"💾 GPU内存: 总计{total_mem:.0f}MB, 可用{available_mem:.0f}MB")
        
        # 测试候选
        candidates = [16, 32, 48, 64, 80, 96, 128, 160, 192, 256]
        
        # 基于可用内存过滤候选
        max_candidate = min(256, int(available_mem / 20))  # 粗略估算
        valid_candidates = [bs for bs in candidates if bs <= max_candidate]
        
        if not quiet:
            print(f"🎯 测试候选: {valid_candidates}")
            print(f"⏰ 超时设置: {self.timeout_seconds}s")
            print("=" * 60)
        
        results = []
        peak_samples_per_sec = 0
        declining_count = 0
        
        for batch_size in valid_candidates:
            result = self.test_batch_with_timeout(batch_size, model)
            
            if result is not None:
                results.append(result)
                current_samples_per_sec = result['samples_per_sec']
                
                # 早停检测
                if current_samples_per_sec > peak_samples_per_sec:
                    peak_samples_per_sec = current_samples_per_sec
                    declining_count = 0
                else:
                    declining_count += 1
                    decline_ratio = (peak_samples_per_sec - current_samples_per_sec) / peak_samples_per_sec
                    
                    # 更敏感的早停
                    if decline_ratio > 0.10 or declining_count >= 2:
                        if not quiet:
                            print(f"🛑 早停: 性能下降{decline_ratio*100:.1f}% 或连续{declining_count}次下降")
                        break
                
                # 时间异常检测
                if result['time_variance'] > 0.6:
                    if not quiet:
                        print(f"🛑 早停: 时间方差过大({result['time_variance']:.2f})")
                    break
                    
                # 内存压力检测
                if result['peak_memory_mb'] / total_mem > 0.85:
                    if not quiet:
                        print(f"🛑 早停: 内存使用过高({result['peak_memory_mb']/total_mem*100:.1f}%)")
                    break
                         else:
                 # 测试失败或超时，立即停止
                 if not quiet:
                     print(f"🛑 停止测试 (batch_size={batch_size}失败或超时)")
                 break
            
            time.sleep(0.2)  # 短暂休息
        
        # 清理
        del model
        torch.cuda.empty_cache()
        
        # 选择最优batch size
        if not results:
            if not quiet:
                print("❌ 没有成功结果，使用默认值32")
            return 32
        
        # 综合评分
        best_throughput = max(results, key=lambda x: x['samples_per_sec'])
        
        efficiency_scores = []
        for result in results:
            throughput_score = result['samples_per_sec'] / best_throughput['samples_per_sec']
            memory_score = 1.0 - (result['peak_memory_mb'] / total_mem)
            stability_score = 1.0 - min(result['time_variance'], 1.0)
            
            overall_score = throughput_score * 0.6 + memory_score * 0.25 + stability_score * 0.15
            efficiency_scores.append((result, overall_score))
        
        best_overall = max(efficiency_scores, key=lambda x: x[1])
        optimal_batch_size = best_overall[0]['batch_size']
        
        if not quiet:
            print(f"\n📈 测试结果:")
            print("=" * 60)
            print(f"{'Batch':<6} {'时间':<8} {'速度':<10} {'内存':<8} {'稳定性':<8}")
            print("-" * 50)
            
            for result in results:
                stability = "好" if result['time_variance'] < 0.2 else "一般" if result['time_variance'] < 0.5 else "差"
                marker = ""
                if result == best_throughput:
                    marker += "🚀"
                if result == best_overall[0]:
                    marker += "⚡"
                
                print(f"{result['batch_size']:<6} {result['avg_time']:<8.2f} "
                      f"{result['samples_per_sec']:<10.0f} {result['peak_memory_mb']:<8.0f} "
                      f"{stability:<8} {marker}")
            
            print(f"\n🏆 最优选择: batch_size={optimal_batch_size}")
            print(f"   (评分: {best_overall[1]:.3f})")
        
        return optimal_batch_size

def find_optimal_batch_size(quiet: bool = False) -> int:
    """快速找到最优batch size"""
    optimizer = TimeoutBatchOptimizer(timeout_seconds=12.0)
    return optimizer.find_optimal_batch_size(quiet=quiet)

if __name__ == "__main__":
    find_optimal_batch_size(quiet=False) 