"""
\defgroup group_minimal_monitor Minimal Monitor
\ingroup core
Minimal Monitor module for NeuroExapt framework.
"""


import time
import logging
import os
from typing import Dict, Any, List, Optional
from datetime import datetime

class MinimalMonitor:
    """
    极简性能监控系统
    专注于核心功能，去除所有可能导致性能问题的复杂特性
    """
    
    def __init__(self, log_dir: str = "minimal_logs"):
        self.log_dir = log_dir
        self.start_time = time.time()
        self.timers: Dict[str, float] = {}
        self.counters: Dict[str, int] = {}
        self.stats: Dict[str, List[float]] = {}
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 设置简单的日志记录器
        self.logger = self._setup_logger()
        
        self.logger.info("MinimalMonitor initialized")
    
    def _setup_logger(self) -> logging.Logger:
        """设置简单的日志记录器"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = os.path.join(self.log_dir, f"minimal_{timestamp}.log")
        
        logger = logging.getLogger('minimal_monitor')
        logger.setLevel(logging.INFO)
        
        # 清除已有的处理器
        logger.handlers.clear()
        
        # 只保留文件处理器
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 简单的格式
        formatter = logging.Formatter('%(asctime)s | %(message)s')
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        
        return logger
    
    def start_timer(self, name: str):
        """开始计时"""
        self.timers[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """结束计时并返回耗时"""
        if name not in self.timers:
            return 0.0
        
        elapsed = time.time() - self.timers[name]
        del self.timers[name]
        
        # 记录统计
        if name not in self.stats:
            self.stats[name] = []
        self.stats[name].append(elapsed)
        
        return elapsed
    
    def count(self, name: str, value: int = 1):
        """计数器"""
        if name not in self.counters:
            self.counters[name] = 0
        self.counters[name] += value
    
    def log(self, message: str):
        """记录日志"""
        self.logger.info(message)
    
    def log_epoch(self, epoch: int, train_acc: float, valid_acc: float, 
                  train_time: float, valid_time: float, arch_updates: int = 0):
        """记录epoch摘要"""
        self.log(f"Epoch {epoch}: Train={train_acc:.2f}% Valid={valid_acc:.2f}% "
                f"Time={train_time:.1f}s+{valid_time:.1f}s ArchUpdates={arch_updates}")
    
    def log_step(self, epoch: int, step: int, loss: float, acc: float, 
                 step_time: float, arch_time: float = 0.0):
        """记录训练步骤"""
        arch_info = f" Arch={arch_time:.3f}s" if arch_time > 0 else ""
        self.log(f"[{epoch}][{step}] Loss={loss:.4f} Acc={acc:.2f}% "
                f"Time={step_time:.3f}s{arch_info}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取简单的统计信息"""
        stats = {
            'total_runtime': time.time() - self.start_time,
            'counters': self.counters.copy(),
            'timing_averages': {}
        }
        
        # 计算平均时间
        for name, times in self.stats.items():
            if times:
                stats['timing_averages'][name] = {
                    'avg': sum(times) / len(times),
                    'count': len(times),
                    'total': sum(times)
                }
        
        return stats
    
    def cleanup(self):
        """清理资源"""
        stats = self.get_stats()
        self.log(f"Final stats: {stats}")
        
        # 关闭日志处理器
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler) 