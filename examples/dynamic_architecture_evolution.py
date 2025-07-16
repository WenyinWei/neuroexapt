#!/usr/bin/env python3
"""
动态架构演进系统 - 真正的架构自发增长

🧬 核心能力：
1. 动态增加/移除层（深度演进）
2. 动态调整通道数（宽度演进） 
3. 动态增减分支（拓扑演进）
4. 基于性能的智能架构决策
5. 自动形状匹配和参数迁移

🎯 目标：让神经网络自发寻找最适合的架构！
"""

import argparse
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import json
import copy
from datetime import datetime
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuroexapt.core import CheckpointManager, get_checkpoint_manager

class EvolvableBlock(nn.Module):
    """可演进的基础块"""
    
    def __init__(self, in_channels, out_channels, block_id, stride=1):
        super(EvolvableBlock, self).__init__()
        
        self.block_id = block_id
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        
        # 可扩展的操作列表
        self.operations = nn.ModuleList()
        self.skip_connections = nn.ModuleList()
        
        # 初始基础操作
        self._add_basic_operations()
        
        # 架构参数：控制操作选择和跳跃连接
        self.op_weights = nn.Parameter(torch.randn(len(self.operations)))
        self.skip_weights = nn.Parameter(torch.randn(3))  # [no_skip, add, concat]
        
        # 演进历史
        self.evolution_history = []
        
    def _add_basic_operations(self):
        """添加基础操作"""
        # 3x3 Conv
        self.operations.append(nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, 3, 
                     stride=self.stride, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=False)
        ))
        
        # 1x1 Conv（通道调整）
        self.operations.append(nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, 1, 
                     stride=self.stride, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=False)
        ))
        
        # Depthwise Separable
        self.operations.append(nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, 3, 
                     stride=self.stride, padding=1, groups=self.in_channels, bias=False),
            nn.Conv2d(self.in_channels, self.out_channels, 1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=False)
        ))
    
    def forward(self, x, skip_input=None):
        """前向传播"""
        # 计算所有操作的加权输出
        op_weights = F.softmax(self.op_weights, dim=0)
        output = sum(w * op(x) for w, op in zip(op_weights, self.operations))
        
        # 处理跳跃连接
        if skip_input is not None and skip_input.shape[1] == output.shape[1]:
            skip_weights = F.softmax(self.skip_weights, dim=0)
            
            # 形状匹配
            if skip_input.shape[2:] != output.shape[2:]:
                skip_input = F.adaptive_avg_pool2d(skip_input, output.shape[2:])
            
            # 跳跃连接模式
            if skip_weights[1] > 0.5:  # Add
                output = output + skip_weights[1] * skip_input
            elif skip_weights[2] > 0.3:  # Concat（需要通道调整）
                if skip_input.shape[1] <= output.shape[1]:
                    padding = output.shape[1] - skip_input.shape[1]
                    skip_input = F.pad(skip_input, (0, 0, 0, 0, 0, padding))
                    output = output + skip_weights[2] * skip_input
        
        return output
    
    def add_operation(self, operation):
        """动态添加新操作"""
        self.operations.append(operation)
        
        # 扩展权重参数
        old_weights = self.op_weights.data
        new_weights = torch.randn(len(self.operations))
        new_weights[:-1] = old_weights
        new_weights[-1] = old_weights.mean()  # 初始化为平均值
        
        self.op_weights = nn.Parameter(new_weights)
        
        self.evolution_history.append({
            'action': 'add_operation',
            'operation_type': str(type(operation)),
            'timestamp': time.time()
        })
        
        print(f"🧬 Block {self.block_id}: Added new operation, total={len(self.operations)}")
    
    def get_dominant_operation(self):
        """获取主导操作"""
        with torch.no_grad():
            weights = F.softmax(self.op_weights, dim=0)
            dominant_idx = torch.argmax(weights).item()
            return {
                'index': dominant_idx,
                'weight': weights[dominant_idx].item(),
                'entropy': (-weights * torch.log(weights + 1e-8)).sum().item()
            }

class DynamicArchitecture(nn.Module):
    """动态演进架构"""
    
    def __init__(self, initial_channels=16, num_classes=10):
        super(DynamicArchitecture, self).__init__()
        
        self.initial_channels = initial_channels
        self.num_classes = num_classes
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, initial_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(initial_channels),
            nn.ReLU(inplace=False)
        )
        
        # 动态块列表
        self.blocks = nn.ModuleList()
        
        # 初始架构：3个基础块
        current_channels = initial_channels
        for i in range(3):
            stride = 2 if i > 0 else 1
            out_channels = current_channels * (2 if i > 0 else 1)
            
            block = EvolvableBlock(current_channels, out_channels, i, stride)
            self.blocks.append(block)
            current_channels = out_channels
        
        # 分类头
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(current_channels, num_classes)
        
        # 演进状态
        self.evolution_stats = {
            'total_evolutions': 0,
            'depth_changes': 0,
            'channel_changes': 0,
            'operation_additions': 0
        }
        
        print(f"🏗️ Dynamic Architecture initialized:")
        print(f"   Initial blocks: {len(self.blocks)}")
        print(f"   Initial channels: {initial_channels}")
        print(f"   Current channels: {current_channels}")
    
    def forward(self, x):
        """前向传播"""
        x = self.stem(x)
        
        skip_inputs = [None]  # 用于跳跃连接
        
        for i, block in enumerate(self.blocks):
            skip_input = skip_inputs[-2] if len(skip_inputs) >= 2 else None
            x = block(x, skip_input)
            skip_inputs.append(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
    
    def add_depth(self, position=None):
        """增加网络深度"""
        if position is None:
            position = len(self.blocks)  # 在末尾添加
        
        # 确定新块的通道数
        if position == 0:
            in_channels = self.initial_channels
            out_channels = self.initial_channels
        else:
            prev_block = self.blocks[position - 1]
            in_channels = prev_block.out_channels
            out_channels = in_channels
        
        # 创建新块
        new_block = EvolvableBlock(in_channels, out_channels, 
                                 f"evolved_{len(self.blocks)}", stride=1)
        
        # 插入新块
        self.blocks.insert(position, new_block)
        
        # 更新分类器（如果需要）
        if position == len(self.blocks) - 1:
            old_classifier = self.classifier
            self.classifier = nn.Linear(out_channels, self.num_classes)
            
            # 参数迁移
            with torch.no_grad():
                if old_classifier.weight.shape == self.classifier.weight.shape:
                    self.classifier.weight.copy_(old_classifier.weight)
                    self.classifier.bias.copy_(old_classifier.bias)
        
        self.evolution_stats['depth_changes'] += 1
        self.evolution_stats['total_evolutions'] += 1
        
        print(f"🧬 DEPTH EVOLUTION: Added block at position {position}")
        print(f"   New depth: {len(self.blocks)} blocks")
        return True
    
    def expand_channels(self, block_idx, factor=1.5):
        """扩展指定块的通道数"""
        if block_idx >= len(self.blocks):
            return False
        
        block = self.blocks[block_idx]
        old_out_channels = block.out_channels
        new_out_channels = int(old_out_channels * factor)
        
        # 创建新的块
        new_block = EvolvableBlock(
            block.in_channels, new_out_channels, 
            block.block_id, block.stride
        )
        
        # 参数迁移：复制现有操作的权重
        with torch.no_grad():
            for i, (old_op, new_op) in enumerate(zip(block.operations, new_block.operations)):
                for old_param, new_param in zip(old_op.parameters(), new_op.parameters()):
                    if old_param.shape == new_param.shape:
                        new_param.copy_(old_param)
                    elif len(old_param.shape) == 4:  # Conv权重
                        min_out = min(old_param.shape[0], new_param.shape[0])
                        min_in = min(old_param.shape[1], new_param.shape[1])
                        new_param[:min_out, :min_in] = old_param[:min_out, :min_in]
                    elif len(old_param.shape) == 1:  # BN权重/偏置
                        min_dim = min(old_param.shape[0], new_param.shape[0])
                        new_param[:min_dim] = old_param[:min_dim]
            
            # 复制架构参数
            new_block.op_weights.copy_(block.op_weights)
            new_block.skip_weights.copy_(block.skip_weights)
        
        # 替换块
        self.blocks[block_idx] = new_block
        
        # 更新后续块的输入通道数
        self._update_subsequent_blocks(block_idx, new_out_channels)
        
        self.evolution_stats['channel_changes'] += 1
        self.evolution_stats['total_evolutions'] += 1
        
        print(f"🧬 CHANNEL EXPANSION: Block {block_idx}")
        print(f"   Channels: {old_out_channels} → {new_out_channels}")
        return True
    
    def _update_subsequent_blocks(self, start_idx, new_channels):
        """更新后续块的输入通道数"""
        for i in range(start_idx + 1, len(self.blocks)):
            old_block = self.blocks[i]
            
            # 创建新块，更新输入通道数
            new_block = EvolvableBlock(
                new_channels, old_block.out_channels,
                old_block.block_id, old_block.stride
            )
            
            # 参数迁移（尽力而为）
            with torch.no_grad():
                try:
                    new_block.op_weights.copy_(old_block.op_weights)
                    new_block.skip_weights.copy_(old_block.skip_weights)
                except:
                    pass  # 形状不匹配时使用默认初始化
            
            self.blocks[i] = new_block
            new_channels = old_block.out_channels
    
    def add_advanced_operation(self, block_idx):
        """为指定块添加高级操作"""
        if block_idx >= len(self.blocks):
            return False
        
        block = self.blocks[block_idx]
        
        # 随机选择一种高级操作
        advanced_ops = [
            # 5x5 Conv
            nn.Sequential(
                nn.Conv2d(block.in_channels, block.out_channels, 5, 
                         stride=block.stride, padding=2, bias=False),
                nn.BatchNorm2d(block.out_channels),
                nn.ReLU(inplace=False)
            ),
            # 7x7 Conv
            nn.Sequential(
                nn.Conv2d(block.in_channels, block.out_channels, 7, 
                         stride=block.stride, padding=3, bias=False),
                nn.BatchNorm2d(block.out_channels),
                nn.ReLU(inplace=False)
            ),
            # Dilated Conv
            nn.Sequential(
                nn.Conv2d(block.in_channels, block.out_channels, 3, 
                         stride=block.stride, padding=2, dilation=2, bias=False),
                nn.BatchNorm2d(block.out_channels),
                nn.ReLU(inplace=False)
            ),
            # Grouped Conv
            nn.Sequential(
                nn.Conv2d(block.in_channels, block.out_channels, 3, 
                         stride=block.stride, padding=1, 
                         groups=min(block.in_channels, 4), bias=False),
                nn.BatchNorm2d(block.out_channels),
                nn.ReLU(inplace=False)
            )
        ]
        
        # 随机选择并添加
        selected_op = np.random.choice(advanced_ops)
        block.add_operation(selected_op)
        
        self.evolution_stats['operation_additions'] += 1
        self.evolution_stats['total_evolutions'] += 1
        
        return True
    
    def get_architecture_summary(self):
        """获取架构摘要"""
        summary = {
            'depth': len(self.blocks),
            'evolution_stats': self.evolution_stats,
            'blocks': []
        }
        
        for i, block in enumerate(self.blocks):
            block_info = {
                'id': block.block_id,
                'in_channels': block.in_channels,
                'out_channels': block.out_channels,
                'num_operations': len(block.operations),
                'dominant_op': block.get_dominant_operation()
            }
            summary['blocks'].append(block_info)
        
        return summary

class EvolutionController:
    """演进控制器 - 基于性能指导架构演进"""
    
    def __init__(self):
        self.performance_history = []
        self.evolution_decisions = []
        
        # 演进策略参数
        self.patience = 3  # 性能停滞轮数
        self.improvement_threshold = 0.5  # 改进阈值(%)
        self.evolution_probability = 0.3  # 演进概率
        
    def should_evolve(self, current_accuracy, epoch):
        """判断是否应该进行架构演进"""
        self.performance_history.append(current_accuracy)
        
        # 至少训练5个epoch后再考虑演进
        if len(self.performance_history) < 5:
            return False
        
        # 检查性能停滞
        recent_performance = self.performance_history[-self.patience:]
        max_recent = max(recent_performance)
        min_recent = min(recent_performance)
        
        improvement = max_recent - min_recent
        
        # 性能停滞且随机触发
        should_evolve = (improvement < self.improvement_threshold and 
                        np.random.random() < self.evolution_probability)
        
        if should_evolve:
            print(f"🧬 EVOLUTION TRIGGER at epoch {epoch}")
            print(f"   Recent improvement: {improvement:.2f}%")
            print(f"   Performance plateau detected")
        
        return should_evolve
    
    def select_evolution_strategy(self, model, current_accuracy):
        """选择演进策略"""
        strategies = []
        
        # 基于当前性能选择策略
        if current_accuracy < 40:
            # 低性能：增加深度和宽度
            strategies.extend(['add_depth', 'expand_channels'] * 2)
            strategies.append('add_operation')
        elif current_accuracy < 70:
            # 中等性能：平衡增长
            strategies.extend(['add_depth', 'expand_channels', 'add_operation'])
        else:
            # 高性能：精细调优
            strategies.extend(['add_operation'] * 2)
            strategies.append('expand_channels')
        
        return np.random.choice(strategies)
    
    def execute_evolution(self, model, strategy):
        """执行演进策略"""
        success = False
        
        try:
            if strategy == 'add_depth':
                # 在网络中间添加新层
                position = np.random.randint(1, len(model.blocks))
                success = model.add_depth(position)
                
            elif strategy == 'expand_channels':
                # 扩展随机块的通道数
                block_idx = np.random.randint(0, len(model.blocks))
                factor = np.random.uniform(1.2, 1.8)
                success = model.expand_channels(block_idx, factor)
                
            elif strategy == 'add_operation':
                # 为随机块添加高级操作
                block_idx = np.random.randint(0, len(model.blocks))
                success = model.add_advanced_operation(block_idx)
            
            if success:
                decision = {
                    'strategy': strategy,
                    'timestamp': time.time(),
                    'model_depth': len(model.blocks),
                    'total_params': sum(p.numel() for p in model.parameters())
                }
                self.evolution_decisions.append(decision)
                
                print(f"✅ Evolution executed: {strategy}")
                print(f"   Current depth: {len(model.blocks)}")
                print(f"   Total parameters: {decision['total_params']:,}")
                
        except Exception as e:
            print(f"❌ Evolution failed: {e}")
            success = False
        
        return success

class DynamicArchTrainer:
    """动态架构训练器"""
    
    def __init__(self, experiment_name="dynamic_arch"):
        self.experiment_name = experiment_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 检查点管理
        self.checkpoint_manager = get_checkpoint_manager(
            "./dynamic_arch_checkpoints", experiment_name
        )
        
        # 组件
        self.model = None
        self.evolution_controller = EvolutionController()
        self.weight_optimizer = None
        self.arch_optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        
        # 训练状态
        self.best_accuracy = 0.0
        self.training_stats = []
        
        print(f"🚀 Dynamic Architecture Trainer initialized")
        print(f"🔧 Device: {self.device}")
    
    def setup_data(self, batch_size=96):
        """设置数据"""
        print("📊 Setting up CIFAR-10...")
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        train_dataset = torchvision.datasets.CIFAR10(
            './data', train=True, download=True, transform=transform_train
        )
        test_dataset = torchvision.datasets.CIFAR10(
            './data', train=False, transform=transform_test
        )
        
        # 分割训练数据
        train_size = int(0.8 * len(train_dataset))
        valid_size = len(train_dataset) - train_size
        train_subset, valid_subset = random_split(train_dataset, [train_size, valid_size])
        
        self.train_loader = DataLoader(train_subset, batch_size=batch_size, 
                                     shuffle=True, num_workers=2, pin_memory=True)
        self.valid_loader = DataLoader(valid_subset, batch_size=batch_size, 
                                     shuffle=False, num_workers=2, pin_memory=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                                    shuffle=False, num_workers=2, pin_memory=True)
        
        print(f"✅ Data ready: {len(train_subset)} train, {len(valid_subset)} valid, {len(test_dataset)} test")
    
    def setup_model(self, initial_channels=16):
        """设置模型"""
        print(f"🏗️ Creating Dynamic Architecture: C={initial_channels}")
        
        self.model = DynamicArchitecture(initial_channels=initial_channels).to(self.device)
        
        # 优化器
        self.weight_optimizer = optim.SGD(
            [p for p in self.model.parameters() if not any(
                'op_weights' in n or 'skip_weights' in n 
                for n, _ in self.model.named_parameters() if p is _[1]
            )],
            lr=0.025, momentum=0.9, weight_decay=3e-4
        )
        
        # 架构参数优化器
        arch_params = []
        for block in self.model.blocks:
            arch_params.extend([block.op_weights, block.skip_weights])
        
        self.arch_optimizer = optim.Adam(arch_params, lr=3e-4, weight_decay=1e-3)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"📊 Total parameters: {total_params:,}")
    
    def train_epoch(self, epoch, mode='weight'):
        """训练一个epoch"""
        self.model.train()
        
        if mode == 'weight':
            # 权重训练：冻结架构参数
            for block in self.model.blocks:
                block.op_weights.requires_grad = False
                block.skip_weights.requires_grad = False
            optimizer = self.weight_optimizer
            data_loader = self.train_loader
            desc = f"🔧 E{epoch:02d} Weight Training"
        else:
            # 架构训练：冻结权重参数
            for p in self.model.parameters():
                p.requires_grad = False
            for block in self.model.blocks:
                block.op_weights.requires_grad = True
                block.skip_weights.requires_grad = True
            optimizer = self.arch_optimizer
            data_loader = self.valid_loader
            desc = f"🧬 E{epoch:02d} Architecture Search"
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(data_loader, desc=desc)
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            pbar.set_postfix({
                'Loss': f'{total_loss/(batch_idx+1):.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        # 恢复梯度状态
        for p in self.model.parameters():
            p.requires_grad = True
        
        return total_loss/len(data_loader), 100.*correct/total
    
    def validate(self):
        """验证"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        return total_loss/len(self.test_loader), 100.*correct/total
    
    def train(self, epochs=80, initial_channels=16, batch_size=96):
        """主训练流程"""
        print(f"🎯 DYNAMIC ARCHITECTURE EVOLUTION TRAINING")
        print(f"📊 Config: epochs={epochs}, channels={initial_channels}, batch_size={batch_size}")
        
        start_time = time.time()
        
        # 设置
        self.setup_data(batch_size)
        self.setup_model(initial_channels)
        
        # 训练循环
        for epoch in range(epochs):
            print(f"\n{'='*80}")
            print(f"Epoch {epoch+1}/{epochs}")
            
            # 交替训练模式
            if epoch % 4 == 3:  # 每4个epoch进行1次架构搜索
                train_loss, train_acc = self.train_epoch(epoch, 'arch')
                mode = "Architecture Search"
            else:
                train_loss, train_acc = self.train_epoch(epoch, 'weight')
                mode = "Weight Training"
            
            # 验证
            valid_loss, valid_acc = self.validate()
            
            # 记录统计
            stats = {
                'epoch': epoch,
                'mode': mode,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'valid_loss': valid_loss,
                'valid_acc': valid_acc,
                'architecture': self.model.get_architecture_summary()
            }
            self.training_stats.append(stats)
            
            # 输出结果
            print(f"📊 {mode}")
            print(f"   Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
            print(f"   Valid: Loss={valid_loss:.4f}, Acc={valid_acc:.2f}%")
            print(f"   Best:  {self.best_accuracy:.2f}%")
            
            # 架构演进决策
            if self.evolution_controller.should_evolve(valid_acc, epoch):
                strategy = self.evolution_controller.select_evolution_strategy(
                    self.model, valid_acc
                )
                success = self.evolution_controller.execute_evolution(
                    self.model, strategy
                )
                
                if success:
                    # 重新设置优化器（参数可能改变）
                    self.setup_model(initial_channels)
            
            # 更新最佳性能
            if valid_acc > self.best_accuracy:
                self.best_accuracy = valid_acc
                
                # 保存最佳模型
                try:
                    checkpoint_path = self.checkpoint_manager.save_checkpoint(
                        epoch=epoch,
                        model_state=self.model.state_dict(),
                        optimizer_states={
                            'weight': self.weight_optimizer.state_dict(),
                            'arch': self.arch_optimizer.state_dict()
                        },
                        scheduler_states={},  # 添加缺失的参数
                        training_stats=stats,
                        framework_state={     # 添加缺失的参数
                            'evolution_stats': self.model.evolution_stats,
                            'evolution_controller': {
                                'performance_history': self.evolution_controller.performance_history,
                                'evolution_decisions': self.evolution_controller.evolution_decisions
                            }
                        },
                        performance_metric=valid_acc,
                        architecture_info=self.model.get_architecture_summary()
                    )
                    print(f"💾 New best model saved: {valid_acc:.2f}%")
                except Exception as e:
                    print(f"⚠️ Save failed: {e}")
            
            # 显示当前架构
            if epoch % 10 == 9:
                self._display_architecture()
            
            # 内存清理
            if epoch % 5 == 0:
                torch.cuda.empty_cache()
        
        # 训练完成
        total_time = time.time() - start_time
        self._display_final_results(total_time)
    
    def _display_architecture(self):
        """显示当前架构"""
        summary = self.model.get_architecture_summary()
        print(f"\n🏗️ CURRENT ARCHITECTURE:")
        print(f"   Depth: {summary['depth']} blocks")
        print(f"   Evolution stats: {summary['evolution_stats']}")
        
        for i, block_info in enumerate(summary['blocks']):
            dom_op = block_info['dominant_op']
            print(f"   Block {i}: {block_info['in_channels']}→{block_info['out_channels']} "
                  f"({block_info['num_operations']} ops, dominant: {dom_op['weight']:.3f})")
    
    def _display_final_results(self, total_time):
        """显示最终结果"""
        print(f"\n🎉 DYNAMIC EVOLUTION COMPLETED!")
        print(f"⏱️  Total time: {total_time/60:.1f} minutes")
        print(f"🏆 Best accuracy: {self.best_accuracy:.2f}%")
        
        final_summary = self.model.get_architecture_summary()
        print(f"\n🧬 FINAL EVOLVED ARCHITECTURE:")
        print(f"   Final depth: {final_summary['depth']} blocks")
        print(f"   Total evolutions: {final_summary['evolution_stats']['total_evolutions']}")
        print(f"   Depth changes: {final_summary['evolution_stats']['depth_changes']}")
        print(f"   Channel changes: {final_summary['evolution_stats']['channel_changes']}")
        print(f"   Operation additions: {final_summary['evolution_stats']['operation_additions']}")
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"   Final parameters: {total_params:,}")

def main():
    parser = argparse.ArgumentParser(description='Dynamic Architecture Evolution Training')
    
    parser.add_argument('--epochs', type=int, default=60, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=96, help='批大小')
    parser.add_argument('--channels', type=int, default=16, help='初始通道数')
    parser.add_argument('--experiment', type=str, default='dynamic_evolution', help='实验名称')
    
    args = parser.parse_args()
    
    print("🧬 DYNAMIC ARCHITECTURE EVOLUTION - REAL ARCHITECTURE CHANGES!")
    print(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Configuration: {vars(args)}")
    
    trainer = DynamicArchTrainer(args.experiment)
    trainer.train(
        epochs=args.epochs,
        initial_channels=args.channels,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main() 