#!/usr/bin/env python3
"""
智能Batch Size优化器

基于用户建议的改进版本：
1. 快速检测GPU内存腾挪状态
2. 理论计算最大batch size上限  
3. 智能退出机制避免长时间等待

特点：
- 监控GPU计算占用率模式识别内存瓶颈
- 基于GPU内存理论计算batch size上限
- 快速识别并跳过内存腾挪状态
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import gc
import sys
import os
import threading
import subprocess
from typing import Optional, Dict, List, Tuple
import psutil
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
import signal

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from neuroexapt.core.model import Network

class GPUMonitor:
    """GPU监控器，检测内存腾挪状态"""
    
    def __init__(self):
        self.monitoring = False
        self.gpu_utilizations = []
        self.memory_utilizations = []
        self.monitor_thread = None
        
    def start_monitoring(self):
        """开始监控GPU状态"""
        self.monitoring = True
        self.gpu_utilizations = []
        self.memory_utilizations = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            try:
                # 使用nvidia-smi获取GPU利用率
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory', 
                     '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=1.0
                )
                
                if result.returncode == 0:
                    gpu_util, mem_util = map(int, result.stdout.strip().split(', '))
                    self.gpu_utilizations.append(gpu_util)
                    self.memory_utilizations.append(mem_util)
                    
                    # 只保留最近20个样本
                    if len(self.gpu_utilizations) > 20:
                        self.gpu_utilizations = self.gpu_utilizations[-20:]
                        self.memory_utilizations = self.memory_utilizations[-20:]
                        
            except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
                pass  # nvidia-smi不可用或超时，继续监控
                
            time.sleep(0.1)  # 100ms采样间隔
    
    def detect_memory_thrashing(self, min_samples=12) -> Tuple[bool, str]:
        """
        检测是否处于内存腾挪状态 (改进版 - 避免短期峰值误判)
        
        特征识别：
        1. GPU利用率持续稳定高位 (避免短期峰值)
        2. 内存利用率大幅且持续波动
        3. GPU高使用率但训练效率明显低下
        
        Returns:
            (是否检测到腾挪, 检测原因)
        """
        if len(self.gpu_utilizations) < min_samples:
            return False, "样本不足"
            
        recent_gpu = self.gpu_utilizations[-min_samples:]
        recent_mem = self.memory_utilizations[-min_samples:]
        
        # 特征1: GPU持续稳定高利用率（>80%），避免短期峰值
        very_high_gpu_count = sum(1 for x in recent_gpu if x > 80)
        sustained_very_high_gpu = very_high_gpu_count >= min_samples * 0.8  # 80%的时间都>80%
        
        # 特征2: 内存利用率持续波动且在高位
        if len(recent_mem) > 1:
            mem_variance = max(recent_mem) - min(recent_mem)
            high_mem_variance = mem_variance > 25  # 提高阈值，避免误判
            
            # 内存利用率持续在危险高位（>90%）
            very_high_mem_count = sum(1 for x in recent_mem if x > 90)
            sustained_very_high_mem = very_high_mem_count >= min_samples * 0.6
            
            # 内存利用率平均值也要很高
            avg_mem = sum(recent_mem) / len(recent_mem)
        else:
            high_mem_variance = False
            sustained_very_high_mem = False
            avg_mem = 0
        
        # 特征3: GPU平均利用率
        avg_gpu = sum(recent_gpu) / len(recent_gpu)
        
        # 更严格的腾挪判断条件
        reasons = []
        is_thrashing = False
        
        # 条件1: 极高GPU使用率 + 内存在危险区域且波动大
        if sustained_very_high_gpu and sustained_very_high_mem and high_mem_variance:
            reasons.append(f"GPU持续>{very_high_gpu_count}/{min_samples}次>80% + 内存危险区域波动{mem_variance:.0f}%")
            is_thrashing = True
        
        # 条件2: 平均GPU很高但内存利用率异常
        elif avg_gpu > 85 and avg_mem > 85 and high_mem_variance:
            reasons.append(f"双高位运行 GPU({avg_gpu:.0f}%) + 内存({avg_mem:.0f}%) + 波动({mem_variance:.0f}%)")
            is_thrashing = True
        
        # 条件3: 极端情况 - GPU和内存都接近满载
        elif avg_gpu > 90 and avg_mem > 95:
            reasons.append(f"系统接近满载 GPU({avg_gpu:.0f}%) + 内存({avg_mem:.0f}%)")
            is_thrashing = True
        
        reason = "; ".join(reasons) if reasons else "正常"
        return is_thrashing, reason
    
    def get_stats(self) -> Dict:
        """获取监控统计信息"""
        if not self.gpu_utilizations:
            return {'gpu_avg': 0, 'mem_avg': 0, 'samples': 0}
            
        return {
            'gpu_avg': sum(self.gpu_utilizations) / len(self.gpu_utilizations),
            'mem_avg': sum(self.memory_utilizations) / len(self.memory_utilizations),
            'gpu_max': max(self.gpu_utilizations),
            'mem_max': max(self.memory_utilizations),
            'samples': len(self.gpu_utilizations)
        }

class BatchSizeCalculator:
    """Batch Size理论计算器"""
    
    @staticmethod
    def get_gpu_memory_info() -> Tuple[int, int, int]:
        """
        获取GPU内存信息
        
        Returns:
            (总内存MB, 可用内存MB, 系统占用MB)
        """
        if not torch.cuda.is_available():
            return 0, 0, 0
            
        # 获取GPU总内存
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024  # MB
        
        # 清理缓存后获取当前内存使用
        torch.cuda.empty_cache()
        current_allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        
        # 估算系统占用（显存的基础占用）
        system_overhead = max(500, total_memory * 0.1)  # 至少500MB或10%
        
        available_memory = total_memory - system_overhead - current_allocated
        
        return int(total_memory), int(available_memory), int(system_overhead + current_allocated)
    
    @staticmethod
    def estimate_model_memory_per_sample(model: nn.Module) -> Tuple[float, float]:
        """
        精确估算模型每个样本的内存占用（MB）
        
        基于实际测试的改进估算：
        - 前向激活内存：基于网络层数和通道数
        - 反向梯度内存：与激活内存相当
        - 参数和优化器状态：固定开销
        
        Returns:
            (每样本内存占用MB, 固定内存占用MB)
        """
        # 模型参数内存 (float32 = 4 bytes)
        param_memory = sum(p.numel() * 4 for p in model.parameters()) / 1024 / 1024
        
        # 梯度内存（与参数同样大小）
        gradient_memory = param_memory
        
        # 优化器状态内存（SGD + momentum）
        optimizer_memory = param_memory  # 动量缓存
        
        # 激活内存精确估算
        # 基于NeuroExapt模型特性：Network with layers=6, C=16
        # 从实际测试中观察到的内存使用模式：
        
        # CIFAR-10 (3, 32, 32) 输入
        input_size = 3 * 32 * 32 * 4 / 1024 / 1024  # MB
        
        # 每层激活内存（经验估算）
        # stem: 3->64 channels, 32x32
        # cell: progressive channel increase + spatial reduction
        
        # 基于观察到的内存增长模式：
        # batch_size=16: 672MB, batch_size=32: 1272MB
        # 差值: 600MB for 16 samples = 37.5MB per sample
        
        # 但这包含了所有内存，需要分离出激活部分
        total_per_sample_observed = 37.5  # MB from actual testing
        
        # 从中减去其他固定内存的分摊
        fixed_memory = param_memory + gradient_memory + optimizer_memory
        activation_per_sample = total_per_sample_observed - (fixed_memory / 64)  # 假设基准batch=64
        
        # 确保合理范围
        activation_per_sample = max(20.0, min(activation_per_sample, 100.0))  # 20-100MB per sample
        
        return activation_per_sample, fixed_memory
    
    @classmethod
    def calculate_max_batch_size(cls, model: nn.Module, safety_margin: float = 0.8) -> int:
        """
        理论计算最大batch size
        
        Args:
            model: 神经网络模型
            safety_margin: 安全边界（0.8表示只使用80%的可用内存）
        """
        total_mem, available_mem, used_mem = cls.get_gpu_memory_info()
        
        if available_mem <= 0:
            return 32  # 保守估计
        
        # 估算内存需求
        per_sample_mem, fixed_mem = cls.estimate_model_memory_per_sample(model)
        
        # 计算可用于batch的内存
        usable_memory = available_mem * safety_margin - fixed_mem
        
        if usable_memory <= 0 or per_sample_mem <= 0:
            return 32  # 保守估计
        
        # 计算理论最大batch size
        max_batch_size = int(usable_memory / per_sample_mem)
        
        # 限制在合理范围内
        max_batch_size = max(16, min(max_batch_size, 512))
        
        # 调整为8的倍数（GPU计算效率更好）
        max_batch_size = (max_batch_size // 8) * 8
        
        return max_batch_size

def create_test_model() -> nn.Module:
    """创建测试用的模型"""
    return Network(
        C=16,
        num_classes=10,
        layers=6,  # 减少层数以加快测试
        potential_layers=2,  # 减少潜在层数
        use_gradient_optimized=True,
        quiet=True
    ).cuda()

def _run_batch_test_core(batch_size: int, model: nn.Module) -> Optional[Dict]:
    """
    核心batch size测试函数（在线程中运行）
    """
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = optim.SGD(model.parameters(), lr=0.025, momentum=0.9, weight_decay=3e-4)
        
        # 创建测试数据
        test_input = torch.randn(batch_size, 3, 32, 32, device='cuda')
        test_target = torch.randint(0, 10, (batch_size,), device='cuda')
        
        # 预热一次
        model.train()
        output = model(test_input)
        loss = criterion(output, target=test_target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        torch.cuda.synchronize()  # 确保预热完成
        
        # 多次测试取平均
        num_runs = 3
        times = []
        
        for i in range(num_runs):
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            model.train()
            output = model(test_input)
            loss = criterion(output, target=test_target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            batch_time = end_time - start_time
            times.append(batch_time)
        
        # 获取内存信息
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        # 计算统计信息
        avg_time = sum(times) / len(times)
        time_std = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
        time_variance = time_std / avg_time if avg_time > 0 else 0
        samples_per_sec = batch_size / avg_time
        
        return {
            'batch_size': batch_size,
            'avg_time': avg_time,
            'time_variance': time_variance,
            'peak_memory_mb': peak_memory_mb,
            'samples_per_sec': samples_per_sec,
            'times': times
        }
        
    except Exception as e:
        return None

def test_batch_with_monitoring(batch_size: int, model: nn.Module, monitor: GPUMonitor, timeout_seconds: float = 15.0) -> Optional[Dict]:
    """
    带监控和超时的batch size测试
    
    使用线程池和超时机制避免内存腾挪时长时间等待
    """
    print(f"测试 batch_size={batch_size:3d}... ", end="", flush=True)
    
    # 开始监控
    monitor.start_monitoring()
    
    try:
        # 使用线程池执行测试，设置超时
        with ThreadPoolExecutor(max_workers=1) as executor:
            # 提交测试任务
            future = executor.submit(_run_batch_test_core, batch_size, model)
            
            try:
                # 等待结果，设置超时
                result = future.result(timeout=timeout_seconds)
                
                if result is not None:
                    # 检查是否有内存腾挪
                    time.sleep(0.3)  # 给监控器收集数据的时间
                    is_thrashing, reason = monitor.detect_memory_thrashing()
                    
                    if is_thrashing:
                        monitor.stop_monitoring()
                        print(f"❌ 内存腾挪 ({reason})")
                        return None
                    else:
                        monitor.stop_monitoring()
                        print(f"✅ {result['avg_time']:.2f}s/batch, {result['samples_per_sec']:4.0f} samples/s, {result['peak_memory_mb']:4.0f}MB")
                        return result
                else:
                    monitor.stop_monitoring()
                    print(f"❌ 测试失败")
                    return None
                    
            except TimeoutError:
                # 超时了，强制取消任务
                future.cancel()
                monitor.stop_monitoring()
                print(f"🕒 超时 (>{timeout_seconds:.0f}s, 可能内存腾挪)")
                
                # 清理CUDA缓存
                torch.cuda.empty_cache()
                return None
                
    except Exception as e:
        monitor.stop_monitoring()
        print(f"❌ 错误: {str(e)}")
        return None
                    monitor.stop_monitoring()
                    print(f"❌ 内存腾挪 (第{i+1}次后: {reason})")
                    return None
            
            # 超时检查
            if batch_time > 15.0:  # 15秒超时
                monitor.stop_monitoring()
                print(f"❌ 超时 ({batch_time:.1f}s)")
                return None
        
        monitor.stop_monitoring()
        
        # 计算结果
        avg_time = sum(times) / len(times)
        samples_per_sec = batch_size / avg_time
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        time_variance = max(times) - min(times) if len(times) > 1 else 0
        
        # 获取GPU监控统计
        monitor_stats = monitor.get_stats()
        
        print(f"✅ {avg_time:.2f}s/batch, {samples_per_sec:5.0f} samples/s, {peak_memory:4.0f}MB")
        if monitor_stats['samples'] > 0:
            print(f"       GPU平均:{monitor_stats['gpu_avg']:.0f}% 内存平均:{monitor_stats['mem_avg']:.0f}%")
        
        # 清理
        del test_input, test_target
        torch.cuda.empty_cache()
        gc.collect()
        
        return {
            'batch_size': batch_size,
            'avg_time': avg_time,
            'samples_per_sec': samples_per_sec,
            'peak_memory_mb': peak_memory,
            'warmup_time': warmup_time,
            'time_variance': time_variance,
            'gpu_stats': monitor_stats
        }
        
    except RuntimeError as e:
        monitor.stop_monitoring()
        if "out of memory" in str(e).lower():
            print("❌ OOM")
        else:
            print(f"❌ Error: {str(e)[:30]}...")
        return None
    except Exception as e:
        monitor.stop_monitoring()
        print(f"❌ Exception: {str(e)[:30]}...")
        return None

def find_optimal_batch_size(quiet: bool = False) -> int:
    """
    静默找到最优batch size
    
    Returns:
        最优的batch size数值
    """
    if not quiet:
        print("🧠 智能Batch Size优化器")
        print("=" * 60)
    
    # 创建测试模型
    if not quiet:
        print("📊 创建测试模型...")
    model = create_test_model()
    
    # 获取GPU内存信息
    total_mem, available_mem, used_mem = BatchSizeCalculator.get_gpu_memory_info()
    if not quiet:
        print(f"💾 GPU内存: 总计{total_mem}MB, 可用{available_mem}MB, 已用{used_mem}MB")
    
    # 理论计算最大batch size
    theoretical_max = BatchSizeCalculator.calculate_max_batch_size(model)
    if not quiet:
        print(f"🧮 理论最大batch size: {theoretical_max}")
    
    # 创建GPU监控器
    monitor = GPUMonitor()
    
    # 智能测试策略 - 扩展候选范围支持各种GPU配置
    candidates = [16, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256, 320, 384, 512]
    
    # 只测试不超过理论最大值的batch size
    valid_candidates = [bs for bs in candidates if bs <= theoretical_max * 1.2]  # 允许20%超出
    
    if not valid_candidates:
        valid_candidates = [16, 32]  # 保守测试
    
    if not quiet:
        print(f"🎯 测试候选: {valid_candidates}")
        print("=" * 60)
    
    results = []
    peak_samples_per_sec = 0  # 记录峰值吞吐量
    declining_count = 0  # 连续下降计数
    last_two_results = []  # 记录最近两次结果，用于趋势判断
    
    for batch_size in valid_candidates:
        result = test_batch_with_monitoring(batch_size, model, monitor)
        
        if result is not None:
            results.append(result)
            current_samples_per_sec = result['samples_per_sec']
            current_time_per_batch = result['avg_time']
            
            # 检测性能下降的多种指标
            should_stop = False
            stop_reason = ""
            
            # 1. 吞吐量下降检测
            if current_samples_per_sec > peak_samples_per_sec:
                peak_samples_per_sec = current_samples_per_sec
                declining_count = 0  # 重置下降计数
            else:
                declining_count += 1
                decline_ratio = (peak_samples_per_sec - current_samples_per_sec) / peak_samples_per_sec
                
                # 更敏感的早停条件
                if decline_ratio > 0.12:  # 下降超过12%就停止
                    should_stop = True
                    stop_reason = f"吞吐量下降{decline_ratio*100:.1f}%"
                elif declining_count >= 2:  # 连续2次下降
                    should_stop = True
                    stop_reason = f"连续{declining_count}次下降"
            
            # 2. 时间剧增检测（内存腾挪的典型表现）
            if len(last_two_results) >= 2:
                recent_avg_time = sum(r['avg_time'] for r in last_two_results) / len(last_two_results)
                time_increase_ratio = (current_time_per_batch - recent_avg_time) / recent_avg_time
                
                if time_increase_ratio > 0.5:  # 时间增长超过50%
                    should_stop = True
                    stop_reason = f"运行时间剧增{time_increase_ratio*100:.1f}%"
            
            # 3. 时间方差检测（不稳定性）
            if result['time_variance'] > 0.8:  # 时间方差过大，说明内存腾挪严重
                should_stop = True
                stop_reason = f"时间方差过大({result['time_variance']:.2f})"
            
            # 4. 内存压力检测
            memory_usage_ratio = result['peak_memory_mb'] / total_mem
            if memory_usage_ratio > 0.85:  # 内存使用超过85%
                should_stop = True
                stop_reason = f"内存压力过大({memory_usage_ratio*100:.1f}%)"
            
            # 执行早停
            if should_stop:
                if not quiet:
                    print(f"🛑 智能早停: {stop_reason}")
                    print(f"   当前: {current_samples_per_sec:.0f} samples/s, {current_time_per_batch:.2f}s/batch")
                    print(f"   峰值: {peak_samples_per_sec:.0f} samples/s")
                    print(f"   跳过剩余更大的batch size测试")
                break
            
            # 更新最近结果记录
            last_two_results.append(result)
            if len(last_two_results) > 2:
                last_two_results.pop(0)
                
        else:
            # 如果失败了，跳过更大的batch size
            if not quiet:
                print(f"⚠️  跳过更大的batch size (当前{batch_size}失败)")
            break
        
        # 短暂休息让GPU冷却
        time.sleep(0.3)  # 进一步减少等待时间
    
    # 清理测试模型
    del model
    torch.cuda.empty_cache()
    
    # 分析结果并返回最佳batch size
    if not results:
        if not quiet:
            print("❌ 没有成功的测试结果，使用默认值32")
        return 32
    
    # 计算综合最佳batch size
    best_throughput = max(results, key=lambda x: x['samples_per_sec'])
    
    efficiency_scores = []
    for result in results:
        # 综合评分：吞吐量 + 内存效率 + 稳定性
        throughput_score = result['samples_per_sec'] / best_throughput['samples_per_sec']
        memory_score = 1.0 - (result['peak_memory_mb'] / total_mem)
        stability_score = 1.0 - min(result['time_variance'], 1.0)
        
        overall_score = throughput_score * 0.5 + memory_score * 0.3 + stability_score * 0.2
        efficiency_scores.append((result, overall_score))
    
    best_overall = max(efficiency_scores, key=lambda x: x[1])
    optimal_batch_size = best_overall[0]['batch_size']
    
    if not quiet:
        print(f"\n📈 测试结果分析:")
        print("=" * 60)
        print(f"{'Batch':<6} {'时间':<8} {'速度':<10} {'内存':<8} {'效率':<6} {'稳定性':<8}")
        print("-" * 60)
        
        for result in results:
            efficiency = result['samples_per_sec'] / result['peak_memory_mb'] * 1000
            stability = "好" if result['time_variance'] < 0.2 else "一般" if result['time_variance'] < 0.5 else "差"
            
            marker = ""
            if result == best_throughput:
                marker += "🚀"
            if result == best_overall[0]:
                marker += "⚡"
            
            print(f"{result['batch_size']:<6} {result['avg_time']:<8.2f} "
                  f"{result['samples_per_sec']:<10.0f} {result['peak_memory_mb']:<8.0f} "
                  f"{efficiency:<6.1f} {stability:<8} {marker}")
        
        print(f"\n🏆 最优选择: batch_size={optimal_batch_size}")
        print(f"   最高吞吐量: batch_size={best_throughput['batch_size']} "
              f"({best_throughput['samples_per_sec']:.0f} samples/s)")
        print(f"   综合最佳: batch_size={optimal_batch_size} "
              f"(评分: {best_overall[1]:.3f})")
    
    return optimal_batch_size

def intelligent_batch_size_optimization():
    """智能batch size优化（详细版本）"""
    optimal_batch_size = find_optimal_batch_size(quiet=False)
    return optimal_batch_size

if __name__ == "__main__":
    intelligent_batch_size_optimization() 