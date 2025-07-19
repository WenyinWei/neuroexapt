"""
defgroup group_optimized_architect Optimized Architect
ingroup core
Optimized Architect module for NeuroExapt framework.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import time
import gc
from collections import OrderedDict

from .architect import Architect
from ..utils.performance_monitor import PerformanceMonitor

class OptimizedArchitect(Architect):
    """
    优化的架构搜索器，利用PyTorch的参数优化设施来提高性能
    
    主要优化策略：
    1. 减少架构更新频率
    2. 使用一阶近似替代二阶近似
    3. 梯度累积和批处理
    4. 内存管理优化
    5. 早停机制
    """
    
    def __init__(self, model, args, performance_monitor: Optional[PerformanceMonitor] = None):
        super().__init__(model, args)
        
        # 性能监控
        self.monitor = performance_monitor
        
        # 优化参数
        self.arch_update_freq = getattr(args, 'arch_update_freq', 50)
        self.warmup_epochs = getattr(args, 'warmup_epochs', 5)
        self.use_first_order = getattr(args, 'use_first_order', True)
        self.grad_accumulation_steps = getattr(args, 'grad_accumulation_steps', 1)
        self.arch_early_stop_patience = getattr(args, 'arch_early_stop_patience', 20)
        
        # 训练相关
        self.total_epochs = getattr(args, 'epochs', 50)  # 总训练轮数
        
        # 状态跟踪
        self.current_epoch = 0
        self.arch_step_count = 0
        self.last_arch_loss = float('inf')
        self.arch_loss_history = []
        self.no_improve_count = 0
        
        # 梯度累积缓冲区
        self.accumulated_grads = None
        self.accumulation_count = 0
        
        # 优化器配置
        self.optimizer = self._create_optimized_optimizer(args)
        
        # 记录初始化
        if self.monitor:
            self.monitor.logger.info(f"🚀 OptimizedArchitect initialized:")
            self.monitor.logger.info(f"   Update frequency: every {self.arch_update_freq} steps")
            self.monitor.logger.info(f"   Warmup epochs: {self.warmup_epochs}")
            self.monitor.logger.info(f"   First-order approximation: {self.use_first_order}")
            self.monitor.logger.info(f"   Gradient accumulation: {self.grad_accumulation_steps} steps")
    
    def _create_optimized_optimizer(self, args):
        """创建优化的架构参数优化器"""
        arch_params = list(self.model.arch_parameters())
        
        if len(arch_params) == 0:
            return None
        
        # 使用AdamW优化器，更好的权重衰减
        optimizer = torch.optim.AdamW(
            arch_params,
            lr=args.arch_learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=args.arch_weight_decay,
            amsgrad=True  # 使用AMSGrad变体
        )
        
        return optimizer
    
    def set_epoch(self, epoch: int):
        """设置当前epoch"""
        self.current_epoch = epoch
        
        # 动态调整学习率
        if epoch > self.warmup_epochs:
            self._adjust_learning_rate(epoch)
    
    def _adjust_learning_rate(self, epoch: int):
        """动态调整架构学习率"""
        if self.optimizer is None:
            return
        
        # 基于损失历史调整学习率
        if len(self.arch_loss_history) >= 5:
            recent_losses = self.arch_loss_history[-5:]
            if all(recent_losses[i] >= recent_losses[i+1] for i in range(len(recent_losses)-1)):
                # 损失在下降，保持学习率
                pass
            else:
                # 损失不稳定，减小学习率
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.95
                    
                if self.monitor:
                    self.monitor.logger.info(f"🔧 Architecture learning rate adjusted to {param_group['lr']:.6f}")
    
    def should_update_arch(self) -> bool:
        """判断是否应该更新架构"""
        # 预热期跳过，但要考虑总训练时间
        if self.current_epoch < self.warmup_epochs:
            return False
        
        # 早停检查
        if self.no_improve_count >= self.arch_early_stop_patience:
            return False
        
        # 智能频率控制
        self.arch_step_count += 1
        
        # 根据实际情况动态调整频率
        if self.total_epochs <= 5:
            # 短时间训练，需要更频繁的架构更新
            effective_freq = max(5, self.arch_update_freq // 20)
        elif self.total_epochs <= 10:
            effective_freq = max(10, self.arch_update_freq // 10)
        elif self.total_epochs <= 20:
            effective_freq = max(20, self.arch_update_freq // 5)
        else:
            # 长时间训练，使用原始频率
            effective_freq = self.arch_update_freq
        
        should_update = self.arch_step_count % effective_freq == 0
        
        if should_update and self.monitor and effective_freq != self.arch_update_freq:
            self.monitor.logger.info(f"🔧 Architecture update frequency auto-adjusted: "
                                   f"{self.arch_update_freq} -> {effective_freq} "
                                   f"(step {self.arch_step_count})")
        
        return should_update
    
    def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):
        """优化的架构更新步骤"""
        if not self.should_update_arch():
            if self.monitor:
                self.monitor.log_architecture_step('arch_skip', 0.0)
            return
        
        if self.optimizer is None:
            return
        
        step_start_time = time.time()
        
        try:
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                if self.use_first_order or not unrolled:
                    # 使用高效的一阶近似
                    self._step_first_order(input_valid, target_valid)
                else:
                    # 使用二阶近似（更准确但更慢）
                    self._step_second_order(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
        
        except RuntimeError as e:
            if "out of memory" in str(e):
                if self.monitor:
                    self.monitor.logger.warning(f"🔧 OOM in architecture step, falling back to first-order")
                torch.cuda.empty_cache()
                self._step_first_order(input_valid, target_valid)
            else:
                raise e
        
        step_time = time.time() - step_start_time
        
        if self.monitor:
            self.monitor.log_architecture_step('arch_update', step_time, {
                'method': 'first_order' if self.use_first_order else 'second_order',
                'accumulation_steps': self.grad_accumulation_steps
            })
    
    def _step_first_order(self, input_valid, target_valid):
        """高效的一阶近似更新"""
        if self.optimizer is None:
            return
            
        if self.grad_accumulation_steps <= 1:
            # 直接更新
            self.optimizer.zero_grad()
            loss = self._compute_arch_loss(input_valid, target_valid)
            loss.backward()
            
            # 梯度裁剪
            self._clip_arch_gradients()
            
            self.optimizer.step()
            self._update_arch_loss_history(loss.item())
        else:
            # 梯度累积
            self._step_with_accumulation(input_valid, target_valid)
    
    def _step_second_order(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
        """二阶近似更新（优化版本）"""
        if self.optimizer is None:
            return
            
        self.optimizer.zero_grad()
        
        # 使用内存效率更高的二阶近似
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            # 计算虚拟模型
            virtual_model = self._compute_virtual_model_efficient(input_train, target_train, eta, network_optimizer)
            
            # 计算架构损失
            if self.criterion is None:
                raise ValueError("Criterion not set")
            arch_loss = self.criterion(virtual_model(input_valid), target_valid)
            
            # 计算梯度
            arch_grads = torch.autograd.grad(arch_loss, self.model.arch_parameters(), 
                                           create_graph=False, retain_graph=False)
            
            # 应用梯度
            for param, grad in zip(self.model.arch_parameters(), arch_grads):
                if param.grad is None:
                    param.grad = grad.detach()
                else:
                    param.grad.data.copy_(grad.detach())
        
        # 梯度裁剪
        self._clip_arch_gradients()
        
        self.optimizer.step()
        self._update_arch_loss_history(arch_loss.item())
    
    def _compute_virtual_model_efficient(self, input_train, target_train, eta, network_optimizer):
        """内存效率更高的虚拟模型计算"""
        # 计算当前损失和梯度
        if self.criterion is None:
            raise ValueError("Criterion not set")
        self.model.zero_grad()
        loss = self.criterion(self.model(input_train), target_train)
        
        # 计算权重梯度
        weight_grads = torch.autograd.grad(loss, self.model.parameters(), 
                                         create_graph=False, retain_graph=False)
        
        # 创建虚拟参数
        virtual_params = []
        for param, grad in zip(self.model.parameters(), weight_grads):
            # 简化的权重更新（不考虑动量）
            virtual_param = param - eta * grad
            virtual_params.append(virtual_param)
        
        # 创建虚拟模型
        virtual_model = self._create_virtual_model(virtual_params)
        return virtual_model
    
    def _create_virtual_model(self, virtual_params):
        """创建虚拟模型（共享架构参数）"""
        # 这里需要根据具体的模型结构实现
        # 简化版本：直接返回当前模型
        return self.model
    
    def _step_with_accumulation(self, input_valid, target_valid):
        """使用梯度累积的更新"""
        if self.optimizer is None:
            return
            
        # 计算当前批次的损失
        loss = self._compute_arch_loss(input_valid, target_valid)
        loss = loss / self.grad_accumulation_steps
        
        # 反向传播
        loss.backward()
        
        self.accumulation_count += 1
        
        # 检查是否需要更新
        if self.accumulation_count >= self.grad_accumulation_steps:
            # 梯度裁剪
            self._clip_arch_gradients()
            
            # 更新参数
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # 重置计数器
            self.accumulation_count = 0
            
            # 更新损失历史
            self._update_arch_loss_history(loss.item() * self.grad_accumulation_steps)
    
    def _compute_arch_loss(self, input_valid, target_valid):
        """计算架构损失"""
        if self.criterion is None:
            raise ValueError("Criterion not set")
        logits = self.model(input_valid)
        loss = self.criterion(logits, target_valid)
        return loss
    
    def _clip_arch_gradients(self, max_norm: float = 5.0):
        """架构梯度裁剪"""
        if self.model.arch_parameters():
            arch_params = list(self.model.arch_parameters())
            total_norm = torch.nn.utils.clip_grad_norm_(arch_params, max_norm)
            
            if self.monitor and total_norm > max_norm:
                self.monitor.logger.info(f"🔧 Architecture gradients clipped: {total_norm:.4f} -> {max_norm}")
    
    def _update_arch_loss_history(self, loss_value: float):
        """更新架构损失历史"""
        self.arch_loss_history.append(loss_value)
        
        # 保持历史记录长度
        if len(self.arch_loss_history) > 100:
            self.arch_loss_history = self.arch_loss_history[-100:]
        
        # 早停检查
        if loss_value < self.last_arch_loss:
            self.last_arch_loss = loss_value
            self.no_improve_count = 0
        else:
            self.no_improve_count += 1
    
    def get_arch_statistics(self) -> Dict[str, Any]:
        """获取架构搜索统计信息"""
        stats = {
            'arch_step_count': self.arch_step_count,
            'current_epoch': self.current_epoch,
            'last_arch_loss': self.last_arch_loss,
            'no_improve_count': self.no_improve_count,
            'arch_loss_history_length': len(self.arch_loss_history)
        }
        
        if self.arch_loss_history:
            stats.update({
                'arch_loss_mean': np.mean(self.arch_loss_history[-10:]),
                'arch_loss_std': np.std(self.arch_loss_history[-10:]),
                'arch_loss_trend': self._compute_loss_trend()
            })
        
        return stats
    
    def _compute_loss_trend(self) -> str:
        """计算损失趋势"""
        if len(self.arch_loss_history) < 5:
            return 'insufficient_data'
        
        recent_losses = self.arch_loss_history[-5:]
        trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
        
        if trend < -0.001:
            return 'decreasing'
        elif trend > 0.001:
            return 'increasing'
        else:
            return 'stable'
    
    def cleanup_gradients(self):
        """清理梯度和内存"""
        # 清理模型梯度
        self.model.zero_grad()
        
        # 清理架构参数梯度
        for param in self.model.arch_parameters():
            if param.grad is not None:
                param.grad.data.zero_()
        
        # 清理优化器状态
        if self.optimizer is not None:
            self.optimizer.zero_grad()
        
        # 清理累积梯度
        self.accumulated_grads = None
        self.accumulation_count = 0
        
        # 强制垃圾回收
        gc.collect()
        
        # 清理CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def save_checkpoint(self, filepath: str):
        """保存检查点"""
        checkpoint = {
            'arch_step_count': self.arch_step_count,
            'current_epoch': self.current_epoch,
            'last_arch_loss': self.last_arch_loss,
            'arch_loss_history': self.arch_loss_history,
            'no_improve_count': self.no_improve_count,
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None
        }
        
        torch.save(checkpoint, filepath)
        
        if self.monitor:
            self.monitor.logger.info(f"💾 Architecture checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """加载检查点"""
        checkpoint = torch.load(filepath, map_location='cpu')
        
        self.arch_step_count = checkpoint['arch_step_count']
        self.current_epoch = checkpoint['current_epoch']
        self.last_arch_loss = checkpoint['last_arch_loss']
        self.arch_loss_history = checkpoint['arch_loss_history']
        self.no_improve_count = checkpoint['no_improve_count']
        
        if self.optimizer and checkpoint['optimizer_state_dict']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.monitor:
            self.monitor.logger.info(f"📂 Architecture checkpoint loaded from {filepath}")


class ArchitectureSpaceOptimizer:
    """
    架构空间优化器，复用PyTorch的参数优化思路
    """
    
    def __init__(self, model, optimizer_config: Dict[str, Any], monitor: Optional[PerformanceMonitor] = None):
        self.model = model
        self.monitor = monitor
        self.config = optimizer_config
        
        # 创建参数组
        self.param_groups = self._create_parameter_groups()
        
        # 创建优化器
        self.optimizer = self._create_optimizer()
        
        # 学习率调度器
        self.scheduler = self._create_scheduler()
        
        if self.monitor:
            self.monitor.logger.info(f"🔧 ArchitectureSpaceOptimizer initialized with {len(self.param_groups)} parameter groups")
    
    def _create_parameter_groups(self) -> List[Dict[str, Any]]:
        """创建参数组，为不同类型的架构参数设置不同的学习率"""
        param_groups = []
        
        # 正常单元的架构参数
        normal_params = []
        # 降维单元的架构参数
        reduce_params = []
        # 深度相关的架构参数
        depth_params = []
        
        for name, param in self.model.named_parameters():
            if 'arch' in name.lower() or 'alpha' in name.lower():
                if 'normal' in name.lower():
                    normal_params.append(param)
                elif 'reduce' in name.lower():
                    reduce_params.append(param)
                elif 'depth' in name.lower() or 'gate' in name.lower():
                    depth_params.append(param)
        
        # 为不同类型的参数设置不同的学习率
        if normal_params:
            param_groups.append({
                'params': normal_params,
                'lr': self.config.get('normal_lr', 3e-4),
                'weight_decay': self.config.get('normal_wd', 1e-3),
                'name': 'normal_arch'
            })
        
        if reduce_params:
            param_groups.append({
                'params': reduce_params,
                'lr': self.config.get('reduce_lr', 3e-4),
                'weight_decay': self.config.get('reduce_wd', 1e-3),
                'name': 'reduce_arch'
            })
        
        if depth_params:
            param_groups.append({
                'params': depth_params,
                'lr': self.config.get('depth_lr', 6e-4),
                'weight_decay': self.config.get('depth_wd', 1e-4),
                'name': 'depth_arch'
            })
        
        return param_groups
    
    def _create_optimizer(self):
        """创建优化器"""
        optimizer_type = self.config.get('type', 'adamw')
        
        if optimizer_type.lower() == 'adamw':
            return torch.optim.AdamW(self.param_groups, amsgrad=True)
        elif optimizer_type.lower() == 'adam':
            return torch.optim.Adam(self.param_groups, amsgrad=True)
        elif optimizer_type.lower() == 'sgd':
            return torch.optim.SGD(self.param_groups, momentum=0.9, nesterov=True)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    def _create_scheduler(self):
        """创建学习率调度器"""
        scheduler_type = self.config.get('scheduler', 'cosine')
        
        if scheduler_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config.get('max_epochs', 50),
                eta_min=self.config.get('min_lr', 1e-6)
            )
        elif scheduler_type == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('step_size', 20),
                gamma=self.config.get('gamma', 0.5)
            )
        elif scheduler_type == 'none':
            return None
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    
    def step(self, loss):
        """执行一步优化"""
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        if self.config.get('grad_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for group in self.param_groups for p in group['params']],
                self.config['grad_clip']
            )
        
        self.optimizer.step()
        
        # 更新学习率
        if self.scheduler:
            self.scheduler.step()
    
    def get_lr(self):
        """获取当前学习率"""
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def state_dict(self):
        """获取状态字典"""
        return {
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None
        }
    
    def load_state_dict(self, state_dict):
        """加载状态字典"""
        self.optimizer.load_state_dict(state_dict['optimizer'])
        if self.scheduler and state_dict['scheduler']:
            self.scheduler.load_state_dict(state_dict['scheduler']) 