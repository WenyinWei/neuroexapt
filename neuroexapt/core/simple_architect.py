import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from typing import Optional
import gc

from .architect import Architect
from ..utils.minimal_monitor import MinimalMonitor

class SimpleArchitect(Architect):
    """
    极简架构搜索器
    去除所有复杂的动态调整逻辑，专注于核心功能
    """
    
    def __init__(self, model, args, monitor: Optional[MinimalMonitor] = None):
        super().__init__(model, args)
        
        # 监控
        self.monitor = monitor
        
        # 简化的参数
        self.arch_update_freq = getattr(args, 'arch_update_freq', 50)
        self.warmup_epochs = getattr(args, 'warmup_epochs', 5)
        self.use_first_order = getattr(args, 'use_first_order', True)
        
        # 状态跟踪
        self.current_epoch = 0
        self.step_count = 0
        
        # 创建优化器
        self.optimizer = torch.optim.Adam(
            self.model.arch_parameters(),
            lr=args.arch_learning_rate,
            betas=(0.5, 0.999),
            weight_decay=args.arch_weight_decay
        )
        
        if self.monitor:
            self.monitor.log(f"SimpleArchitect initialized: freq={self.arch_update_freq}, warmup={self.warmup_epochs}")
    
    def set_epoch(self, epoch: int):
        """设置当前epoch"""
        self.current_epoch = epoch
    
    def should_update_arch(self) -> bool:
        """判断是否应该更新架构"""
        # 预热期跳过
        if self.current_epoch < self.warmup_epochs:
            self.step_count += 1  # 仍然增加计数但不更新
            return False
        
        # 简单的频率控制
        self.step_count += 1
        should_update = self.step_count % self.arch_update_freq == 0
        
        if self.monitor:
            self.monitor.log(f"Step {self.step_count}: should_update_arch = {should_update}")
        
        return should_update
    
    def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):
        """执行架构更新步骤"""
        if not self.should_update_arch():
            return
        
        if self.criterion is None:
            return
        
        # 保存更新前的架构参数（用于对比）
        prev_alphas_normal = None
        prev_alphas_reduce = None
        if hasattr(self.model, 'alphas_normal'):
            prev_alphas_normal = self.model.alphas_normal.data.clone()
            prev_alphas_reduce = self.model.alphas_reduce.data.clone()
        
        # 优化内存使用：在架构更新前清理不必要的缓存
        self._pre_arch_update_cleanup()
        
        # 简单的一阶更新
        self.optimizer.zero_grad()
        
        try:
            # 使用梯度检查点减少内存使用
            if hasattr(self.model, 'use_gradient_optimized') and self.model.use_gradient_optimized:
                # 启用梯度优化模式的架构损失计算
                with torch.cuda.amp.autocast():  # 使用混合精度加速
                    loss = self._gradient_optimized_loss(input_valid, target_valid)
            else:
                # 标准架构损失计算
                loss = self.criterion(self.model(input_valid), target_valid)
            
            loss.backward()
            
            # 智能梯度裁剪：根据模型大小动态调整
            total_norm = self._adaptive_gradient_clip()
            
            # 更新参数
            self.optimizer.step()
            
            # 检查架构变化
            if prev_alphas_normal is not None:
                normal_change = torch.norm(self.model.alphas_normal.data - prev_alphas_normal).item()
                reduce_change = torch.norm(self.model.alphas_reduce.data - prev_alphas_reduce).item()
                
                if normal_change > 0.01 or reduce_change > 0.01:
                    print(f"    🔄 Architecture parameters updated!")
                    print(f"       Normal change: {normal_change:.4f}")
                    print(f"       Reduce change: {reduce_change:.4f}")
                    print(f"       Validation loss: {loss.item():.4f}")
                    print(f"       Gradient norm: {total_norm:.4f}")
            
            if self.monitor:
                self.monitor.count('arch_updates')
                self.monitor.log(f"Architecture updated at step {self.step_count}, loss={loss.item():.4f}")
            
            # 架构更新后的清理工作
            self._post_arch_update_cleanup()
            
        except Exception as e:
            if self.monitor:
                self.monitor.log(f"Architecture update failed: {e}")
            print(f"    ❌ Architecture update failed: {e}")
    
    def _pre_arch_update_cleanup(self):
        """架构更新前的内存清理"""
        # 清理MixedOp缓存
        for module in self.model.modules():
            if hasattr(module, 'clear_cache'):
                module.clear_cache()
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _post_arch_update_cleanup(self):
        """架构更新后的清理工作"""
        # 重置MixedOp统计
        for module in self.model.modules():
            if hasattr(module, '_stats'):
                # 重置统计计数器避免累积
                if hasattr(module, '_forward_count'):
                    module._forward_count = 0
    
    def _gradient_optimized_loss(self, input_valid, target_valid):
        """使用梯度优化的损失计算"""
        # 使用gradient checkpointing减少内存使用
        import torch.utils.checkpoint as cp
        
        def forward_wrapper(x):
            return self.model(x)
        
        # 分批处理以减少内存峰值
        batch_size = input_valid.size(0)
        if batch_size > 16:
            # 分成较小的批次
            losses = []
            chunk_size = 8
            for i in range(0, batch_size, chunk_size):
                end_idx = min(i + chunk_size, batch_size)
                chunk_input = input_valid[i:end_idx]
                chunk_target = target_valid[i:end_idx]
                
                chunk_output = cp.checkpoint(forward_wrapper, chunk_input, use_reentrant=False)
                chunk_loss = self.criterion(chunk_output, chunk_target)
                losses.append(chunk_loss * chunk_input.size(0))
            
            # 加权平均
            total_loss = sum(losses) / batch_size
            return total_loss
        else:
            # 小批次直接处理
            output = cp.checkpoint(forward_wrapper, input_valid, use_reentrant=False)
            return self.criterion(output, target_valid)
    
    def _adaptive_gradient_clip(self) -> float:
        """自适应梯度裁剪"""
        total_norm = 0.0
        param_count = 0
        
        for p in self.model.arch_parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += p.numel()
        
        total_norm = total_norm ** 0.5
        
        # 根据参数数量动态调整裁剪阈值
        adaptive_clip = max(1.0, 5.0 * (param_count / 10000) ** 0.5)
        
        if total_norm > adaptive_clip:
            clip_coef = adaptive_clip / (total_norm + 1e-6)
            for p in self.model.arch_parameters():
                if p.grad is not None:
                    p.grad.data.mul_(clip_coef)
        
        return total_norm
    
    def cleanup_gradients(self):
        """清理梯度（简化版本）"""
        try:
            self.model.zero_grad()
            for param in self.model.arch_parameters():
                if param.grad is not None:
                    param.grad.data.zero_()
        except Exception as e:
            if self.monitor:
                self.monitor.log(f"Gradient cleanup error: {e}")
        
        # 避免频繁调用 empty_cache，可能导致死锁
        # gc.collect()
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()
    
    def get_stats(self):
        """获取简单的统计信息"""
        return {
            'step_count': self.step_count,
            'current_epoch': self.current_epoch,
            'arch_update_freq': self.arch_update_freq
        } 