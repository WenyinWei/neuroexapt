"""
\defgroup group_performance_monitor Performance Monitor
\ingroup core
Performance Monitor module for NeuroExapt framework.
"""


import time
import logging
import json
import os
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
import torch
import numpy as np
from datetime import datetime
import threading
from contextlib import contextmanager

class PerformanceMonitor:
    """
    高性能监控系统，详细记录各个环节的耗时
    """
    
    def __init__(self, log_dir: str = "performance_logs", log_level: int = logging.INFO):
        self.log_dir = log_dir
        self.timers = {}
        self.stats = defaultdict(list)
        self.current_timers = {}
        self.start_time = time.time()
        self.thread_local = threading.local()
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 设置日志器
        self.logger = self._setup_logger(log_level)
        
        # 性能历史记录
        self.history = {
            'epoch_times': [],
            'train_times': [],
            'arch_times': [],
            'valid_times': [],
            'gpu_memory': [],
            'cpu_usage': [],
            'throughput': []
        }
        
        # 实时统计
        self.real_time_stats = {
            'total_steps': 0,
            'arch_steps': 0,
            'skipped_arch_steps': 0,
            'current_epoch': 0,
            'best_accuracy': 0.0,
            'last_improvement_epoch': 0
        }
        
        self.logger.info("Performance Monitor initialized")
        self.logger.info(f"Log directory: {log_dir}")
    
    def _setup_logger(self, log_level: int) -> logging.Logger:
        """设置性能日志记录器"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = os.path.join(self.log_dir, f"performance_{timestamp}.log")
        
        # 创建日志记录器
        logger = logging.getLogger('performance_monitor')
        logger.setLevel(log_level)
        
        # 清除已有的处理器
        logger.handlers.clear()
        
        # 文件处理器
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_handler.setLevel(log_level)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 设置格式
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    @contextmanager
    def timer(self, name: str, log_immediately: bool = False):
        """计时器上下文管理器"""
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            self.record_time(name, elapsed)
            if log_immediately:
                self.logger.info(f"⏱️ {name}: {elapsed:.4f}s")
    
    def record_time(self, name: str, elapsed_time: float):
        """记录时间统计"""
        self.stats[name].append(elapsed_time)
        
        # 保持最近1000个记录
        if len(self.stats[name]) > 1000:
            self.stats[name] = self.stats[name][-1000:]
    
    def start_timer(self, name: str):
        """开始计时"""
        self.current_timers[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """结束计时并返回耗时"""
        if name not in self.current_timers:
            self.logger.warning(f"Timer '{name}' not found")
            return 0.0
        
        elapsed = time.time() - self.current_timers[name]
        self.record_time(name, elapsed)
        del self.current_timers[name]
        return elapsed
    
    def log_gpu_memory(self):
        """记录GPU内存使用情况"""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            memory_cached = torch.cuda.memory_cached() / 1024**3
            
            self.history['gpu_memory'].append({
                'allocated': memory_allocated,
                'reserved': memory_reserved,
                'cached': memory_cached,
                'timestamp': time.time()
            })
            
            self.logger.info(f"🔧 GPU Memory: {memory_allocated:.2f}GB allocated, "
                           f"{memory_reserved:.2f}GB reserved, {memory_cached:.2f}GB cached")
    
    def log_model_stats(self, model: torch.nn.Module, model_name: str = "model"):
        """记录模型统计信息"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.logger.info(f"📊 {model_name} - Total params: {total_params:,}, "
                        f"Trainable: {trainable_params:,}")
        
        # 记录参数分布
        param_stats = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_stats[name] = {
                    'shape': list(param.shape),
                    'numel': param.numel(),
                    'mean': param.data.mean().item(),
                    'std': param.data.std().item(),
                    'grad_norm': param.grad.norm().item() if param.grad is not None else 0.0
                }
        
        return param_stats
    
    def log_epoch_summary(self, epoch: int, train_acc: float, valid_acc: float, 
                         train_time: float, valid_time: float, arch_time: float = 0.0):
        """记录epoch摘要"""
        self.real_time_stats['current_epoch'] = epoch
        
        # 更新最佳准确率
        if valid_acc > self.real_time_stats['best_accuracy']:
            self.real_time_stats['best_accuracy'] = valid_acc
            self.real_time_stats['last_improvement_epoch'] = epoch
        
        total_time = train_time + valid_time + arch_time
        
        self.history['epoch_times'].append(total_time)
        self.history['train_times'].append(train_time)
        self.history['valid_times'].append(valid_time)
        self.history['arch_times'].append(arch_time)
        
        # 计算统计信息
        avg_epoch_time = np.mean(self.history['epoch_times'][-10:])
        arch_percentage = (arch_time / total_time * 100) if total_time > 0 else 0
        
        self.logger.info(f"🏆 EPOCH {epoch} SUMMARY:")
        self.logger.info(f"   Train Acc: {train_acc:.2f}% | Valid Acc: {valid_acc:.2f}%")
        self.logger.info(f"   Times: Train {train_time:.1f}s | Valid {valid_time:.1f}s | Arch {arch_time:.1f}s")
        self.logger.info(f"   Total: {total_time:.1f}s | Arch: {arch_percentage:.1f}% | Avg: {avg_epoch_time:.1f}s")
        self.logger.info(f"   Best Acc: {self.real_time_stats['best_accuracy']:.2f}% "
                        f"(Epoch {self.real_time_stats['last_improvement_epoch']})")
    
    def log_architecture_step(self, step_type: str, duration: float, details: Optional[Dict[str, Any]] = None):
        """记录架构搜索步骤"""
        self.real_time_stats['total_steps'] += 1
        
        if step_type == 'arch_update':
            self.real_time_stats['arch_steps'] += 1
        elif step_type == 'arch_skip':
            self.real_time_stats['skipped_arch_steps'] += 1
        
        skip_rate = (self.real_time_stats['skipped_arch_steps'] / 
                    self.real_time_stats['total_steps'] * 100)
        
        self.logger.info(f"🔍 Architecture Step [{step_type}]: {duration:.4f}s "
                        f"(Skip Rate: {skip_rate:.1f}%)")
        
        if details:
            for key, value in details.items():
                self.logger.info(f"   {key}: {value}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """生成性能报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_runtime': time.time() - self.start_time,
            'real_time_stats': self.real_time_stats.copy(),
            'timing_stats': {}
        }
        
        # 统计信息
        for name, times in self.stats.items():
            if times:
                report['timing_stats'][name] = {
                    'count': len(times),
                    'total': sum(times),
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'min': min(times),
                    'max': max(times),
                    'p50': np.percentile(times, 50),
                    'p95': np.percentile(times, 95),
                    'p99': np.percentile(times, 99)
                }
        
        return report
    
    def save_performance_report(self, filename: Optional[str] = None):
        """保存性能报告到文件"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_report_{timestamp}.json"
        
        filepath = os.path.join(self.log_dir, filename)
        report = self.get_performance_report()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"📋 Performance report saved to: {filepath}")
        return filepath
    
    def log_bottleneck_analysis(self):
        """分析性能瓶颈"""
        self.logger.info("🔍 BOTTLENECK ANALYSIS:")
        
        # 找到最耗时的操作
        operation_times = {}
        for name, times in self.stats.items():
            if times:
                operation_times[name] = {
                    'total': sum(times),
                    'avg': np.mean(times),
                    'count': len(times)
                }
        
        # 按总耗时排序
        sorted_operations = sorted(operation_times.items(), 
                                 key=lambda x: x[1]['total'], reverse=True)
        
        total_time = sum(op['total'] for op in operation_times.values())
        
        self.logger.info(f"   Total measured time: {total_time:.2f}s")
        
        for i, (name, stats) in enumerate(sorted_operations[:10]):
            percentage = (stats['total'] / total_time * 100) if total_time > 0 else 0
            self.logger.info(f"   {i+1:2d}. {name}: {stats['total']:.2f}s "
                           f"({percentage:.1f}%) | Avg: {stats['avg']:.4f}s | "
                           f"Count: {stats['count']}")
    
    def suggest_optimizations(self):
        """建议优化策略"""
        self.logger.info("💡 OPTIMIZATION SUGGESTIONS:")
        
        # 分析架构搜索时间占比
        arch_times = self.stats.get('arch_step', [])
        total_times = self.stats.get('epoch_total', [])
        
        if arch_times and total_times:
            arch_percentage = (sum(arch_times) / sum(total_times) * 100)
            if arch_percentage > 30:
                self.logger.info(f"   🔧 Architecture search takes {arch_percentage:.1f}% of time")
                self.logger.info(f"      → Increase arch_update_freq to reduce frequency")
                self.logger.info(f"      → Use first-order approximation (unrolled=False)")
        
        # 检查GPU利用率
        gpu_memory = self.history.get('gpu_memory', [])
        if gpu_memory:
            avg_memory = np.mean([m['allocated'] for m in gpu_memory])
            if avg_memory < 2.0:  # 少于2GB
                self.logger.info(f"   🔧 GPU memory usage is low ({avg_memory:.1f}GB)")
                self.logger.info(f"      → Increase batch size for better GPU utilization")
        
        # 检查训练停滞
        last_improvement = self.real_time_stats.get('last_improvement_epoch', 0)
        current_epoch = self.real_time_stats.get('current_epoch', 0)
        
        if current_epoch - last_improvement > 10:
            self.logger.info(f"   🔧 No improvement for {current_epoch - last_improvement} epochs")
            self.logger.info(f"      → Consider early stopping or learning rate adjustment")
    
    def cleanup(self):
        """清理资源"""
        self.logger.info("🧹 Cleaning up performance monitor...")
        
        # 保存最终报告
        self.save_performance_report()
        
        # 分析瓶颈
        self.log_bottleneck_analysis()
        
        # 建议优化
        self.suggest_optimizations()
        
        # 关闭日志处理器
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler) 