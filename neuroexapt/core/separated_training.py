#!/usr/bin/env python3
"""
"""
defgroup group_separated_training Separated Training
ingroup core
Separated Training module for NeuroExapt framework.
"""


分离训练策略

实现架构参数（alphas）和网络权重参数的分离训练：
1. 大部分epochs训练网络权重参数（固定架构参数）
2. 定期插入架构训练epoch（固定网络权重，训练架构参数）
3. 显著减少总参数量和计算开销
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Iterator
import time

class SeparatedTrainingStrategy:
    """
    分离训练策略类
    
    核心思想：
    - 主要时间训练网络权重（固定架构）→ 快速收敛
    - 偶尔训练架构参数（指导演化方向）→ 渐进优化
    """
    
    def __init__(self, 
                 weight_training_epochs: int = 4,  # 连续训练权重的epoch数
                 arch_training_epochs: int = 1,    # 插入架构训练的epoch数  
                 total_epochs: int = 20,
                 warmup_epochs: int = 5):          # 前几个epoch只训练权重
        
        self.weight_training_epochs = weight_training_epochs
        self.arch_training_epochs = arch_training_epochs  
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        
        # 训练计划
        self.training_schedule = self._create_schedule()
        
        # 统计信息
        self.weight_training_time = 0.0
        self.arch_training_time = 0.0
        
        print(f"🧬 分离训练策略:")
        print(f"   权重训练轮次: {weight_training_epochs}")
        print(f"   架构训练轮次: {arch_training_epochs}")
        print(f"   预热轮次: {warmup_epochs}")
        print(f"   训练计划: {self.get_schedule_summary()}")
    
    def _create_schedule(self) -> List[str]:
        """创建训练计划"""
        schedule = []
        
        # 预热阶段：只训练权重
        for i in range(self.warmup_epochs):
            schedule.append('weight')
        
        # 主训练阶段：交替训练
        remaining_epochs = self.total_epochs - self.warmup_epochs
        cycle_length = self.weight_training_epochs + self.arch_training_epochs
        
        current_epoch = self.warmup_epochs
        while current_epoch < self.total_epochs:
            # 权重训练阶段
            for i in range(self.weight_training_epochs):
                if current_epoch >= self.total_epochs:
                    break
                schedule.append('weight')
                current_epoch += 1
            
            # 架构训练阶段
            for i in range(self.arch_training_epochs):
                if current_epoch >= self.total_epochs:
                    break
                schedule.append('arch')
                current_epoch += 1
        
        return schedule
    
    def get_training_mode(self, epoch: int) -> str:
        """获取当前epoch的训练模式"""
        if epoch < len(self.training_schedule):
            return self.training_schedule[epoch]
        return 'weight'  # 默认训练权重
    
    def get_schedule_summary(self) -> str:
        """获取训练计划摘要"""
        weight_count = self.training_schedule.count('weight')
        arch_count = self.training_schedule.count('arch')
        return f"权重{weight_count}轮 + 架构{arch_count}轮 = 总共{len(self.training_schedule)}轮"
    
    def should_train_weights(self, epoch: int) -> bool:
        """判断是否应该训练网络权重"""
        return self.get_training_mode(epoch) == 'weight'
    
    def should_train_architecture(self, epoch: int) -> bool:
        """判断是否应该训练架构参数"""
        return self.get_training_mode(epoch) == 'arch'

class SeparatedOptimizer:
    """
    分离优化器
    
    分别管理网络权重和架构参数的优化器
    """
    
    def __init__(self, model: nn.Module, weight_lr: float = 0.025, arch_lr: float = 3e-4,
                 weight_momentum: float = 0.9, weight_decay: float = 3e-4):
        
        self.model = model
        
        # 分离参数
        self.weight_params = []
        self.arch_params = []
        
        # 获取架构参数
        if hasattr(model, 'arch_parameters'):
            self.arch_params = list(model.arch_parameters())
        
        # 获取网络权重参数（排除架构参数）
        arch_param_ids = {id(p) for p in self.arch_params}
        for param in model.parameters():
            if id(param) not in arch_param_ids:
                self.weight_params.append(param)
        
        # 创建分离的优化器
        self.weight_optimizer = optim.SGD(
            self.weight_params,
            lr=weight_lr,
            momentum=weight_momentum,
            weight_decay=weight_decay
        )
        
        self.arch_optimizer = optim.Adam(
            self.arch_params,
            lr=arch_lr,
            weight_decay=1e-3
        )
        
        # 统计信息
        print(f"📊 参数统计:")
        print(f"   网络权重参数: {sum(p.numel() for p in self.weight_params):,}")
        print(f"   架构参数: {sum(p.numel() for p in self.arch_params):,}")
        
        weight_params_count = sum(p.numel() for p in self.weight_params)
        arch_params_count = sum(p.numel() for p in self.arch_params)
        total_params = weight_params_count + arch_params_count
        
        if total_params > 0:
            arch_ratio = arch_params_count / total_params * 100
            print(f"   架构参数占比: {arch_ratio:.2f}%")
    
    def zero_grad_weights(self):
        """清零网络权重梯度"""
        self.weight_optimizer.zero_grad()
    
    def zero_grad_arch(self):
        """清零架构参数梯度"""
        self.arch_optimizer.zero_grad()
    
    def step_weights(self):
        """更新网络权重"""
        self.weight_optimizer.step()
    
    def step_arch(self):
        """更新架构参数"""
        self.arch_optimizer.step()
    
    def freeze_arch_params(self):
        """冻结架构参数"""
        for param in self.arch_params:
            param.requires_grad = False
    
    def unfreeze_arch_params(self):
        """解冻架构参数"""
        for param in self.arch_params:
            param.requires_grad = True
    
    def freeze_weight_params(self):
        """冻结网络权重参数"""
        for param in self.weight_params:
            param.requires_grad = False
    
    def unfreeze_weight_params(self):
        """解冻网络权重参数"""
        for param in self.weight_params:
            param.requires_grad = True
    
    def get_lr_schedulers(self):
        """获取学习率调度器"""
        weight_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.weight_optimizer, T_max=20, eta_min=0.001
        )
        arch_scheduler = optim.lr_scheduler.StepLR(
            self.arch_optimizer, step_size=10, gamma=0.5
        )
        return weight_scheduler, arch_scheduler

class SeparatedTrainer:
    """
    分离训练器
    
    实现完整的分离训练逻辑
    """
    
    def __init__(self, model: nn.Module, strategy: SeparatedTrainingStrategy,
                 optimizer: SeparatedOptimizer, criterion: nn.Module):
        
        self.model = model
        self.strategy = strategy
        self.optimizer = optimizer
        self.criterion = criterion
        
        # 获取学习率调度器
        self.weight_scheduler, self.arch_scheduler = optimizer.get_lr_schedulers()
        
        # 统计信息
        self.epoch_stats = {}
    
    def train_epoch_weights(self, train_loader, epoch: int) -> Dict[str, float]:
        """训练网络权重（固定架构参数）"""
        start_time = time.time()
        
        # 冻结架构参数，解冻权重参数
        self.optimizer.freeze_arch_params()
        self.optimizer.unfreeze_weight_params()
        
        self.model.train()
        
        total_loss = 0.0
        total_samples = 0
        correct = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            
            self.optimizer.zero_grad_weights()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.optimizer.weight_params, 5.0)
            
            self.optimizer.step_weights()
            
            # 统计
            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        
        epoch_time = time.time() - start_time
        self.strategy.weight_training_time += epoch_time
        
        avg_loss = total_loss / total_samples
        accuracy = 100.0 * correct / total_samples
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'time': epoch_time
        }
    
    def train_epoch_architecture(self, train_loader, valid_loader, epoch: int) -> Dict[str, float]:
        """训练架构参数（固定网络权重）"""
        start_time = time.time()
        
        # 冻结权重参数，解冻架构参数
        self.optimizer.freeze_weight_params()
        self.optimizer.unfreeze_arch_params()
        
        self.model.train()
        
        total_loss = 0.0
        total_samples = 0
        arch_updates = 0
        
        valid_iter = iter(valid_loader)
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # 获取验证数据用于架构搜索
            try:
                valid_data, valid_target = next(valid_iter)
            except StopIteration:
                valid_iter = iter(valid_loader)
                valid_data, valid_target = next(valid_iter)
            
            data, target = data.cuda(), target.cuda()
            valid_data, valid_target = valid_data.cuda(), valid_target.cuda()
            
            # 架构参数梯度更新
            self.optimizer.zero_grad_arch()
            
            # 在验证数据上评估当前架构
            valid_output = self.model(valid_data)
            arch_loss = self.criterion(valid_output, valid_target)
            
            arch_loss.backward()
            self.optimizer.step_arch()
            
            # 统计
            total_loss += arch_loss.item() * valid_data.size(0)
            total_samples += valid_data.size(0)
            arch_updates += 1
            
            # 定期输出架构搜索进度
            if batch_idx % 50 == 0:
                print(f"    架构搜索步骤 {batch_idx}: 损失={arch_loss.item():.4f}")
        
        epoch_time = time.time() - start_time
        self.strategy.arch_training_time += epoch_time
        
        avg_loss = total_loss / total_samples
        
        return {
            'loss': avg_loss,
            'arch_updates': float(arch_updates),
            'time': epoch_time
        }
    
    def train_epoch(self, train_loader, valid_loader, epoch: int) -> Dict[str, float]:
        """根据策略训练一个epoch"""
        
        # 更新学习率
        if self.strategy.should_train_weights(epoch):
            self.weight_scheduler.step()
        else:
            self.arch_scheduler.step()
        
        # 根据训练模式选择训练方法
        if self.strategy.should_train_weights(epoch):
            print(f"🏋️ Epoch {epoch}: 训练网络权重参数")
            stats = self.train_epoch_weights(train_loader, epoch)
        else:
            print(f"🧬 Epoch {epoch}: 训练架构参数")
            stats = self.train_epoch_architecture(train_loader, valid_loader, epoch)
        
        # 记录统计信息
        self.epoch_stats[epoch] = stats
        
        return stats
    
    def get_final_statistics(self) -> Dict[str, float]:
        """获取最终训练统计"""
        weight_epochs = sum(1 for stats in self.epoch_stats.values() if stats['mode'] == 'weight')
        arch_epochs = sum(1 for stats in self.epoch_stats.values() if stats['mode'] == 'arch')
        
        total_time = self.strategy.weight_training_time + self.strategy.arch_training_time
        
        return {
            'weight_epochs': weight_epochs,
            'arch_epochs': arch_epochs,
            'weight_training_time': self.strategy.weight_training_time,
            'arch_training_time': self.strategy.arch_training_time,
            'total_time': total_time,
            'weight_time_ratio': self.strategy.weight_training_time / total_time if total_time > 0 else 0,
            'time_per_weight_epoch': self.strategy.weight_training_time / weight_epochs if weight_epochs > 0 else 0,
            'time_per_arch_epoch': self.strategy.arch_training_time / arch_epochs if arch_epochs > 0 else 0
        }

def create_separated_training_setup(model: nn.Module, 
                                  weight_training_epochs: int = 4,
                                  arch_training_epochs: int = 1,
                                  total_epochs: int = 20) -> Tuple[SeparatedTrainingStrategy, SeparatedOptimizer, SeparatedTrainer]:
    """
    创建完整的分离训练设置
    
    Returns:
        strategy: 训练策略
        optimizer: 分离优化器  
        trainer: 训练器
    """
    
    # 创建训练策略
    strategy = SeparatedTrainingStrategy(
        weight_training_epochs=weight_training_epochs,
        arch_training_epochs=arch_training_epochs,
        total_epochs=total_epochs
    )
    
    # 创建分离优化器
    optimizer = SeparatedOptimizer(model)
    
    # 创建损失函数
    criterion = nn.CrossEntropyLoss().cuda()
    
    # 创建训练器
    trainer = SeparatedTrainer(model, strategy, optimizer, criterion)
    
    return strategy, optimizer, trainer

if __name__ == "__main__":
    # 测试分离训练策略
    strategy = SeparatedTrainingStrategy(
        weight_training_epochs=4,
        arch_training_epochs=1, 
        total_epochs=20
    )
    
    print("训练计划:")
    for epoch in range(20):
        mode = strategy.get_training_mode(epoch)
        print(f"Epoch {epoch}: {mode}") 