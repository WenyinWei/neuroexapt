#!/usr/bin/env python3
"""
自适应神经网络生长系统 - 真正的结构变化

🌱 核心理念：从小网络开始，基于性能需求真正生长
- 不是搜索操作，而是增加结构
- 层数真正从3→4→5→6层增长
- 参数量显著增加：1万→3万→8万→20万
- 每次生长都是结构性的改变

🎯 生长策略：
1. 深度生长：在网络中插入新的卷积层
2. 宽度生长：扩展现有层的通道数
3. 分支生长：增加并行处理分支
4. 智能决策：基于性能瓶颈选择生长方式
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import time
import logging
from datetime import datetime
import json
import os
import sys
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuroexapt.core import CheckpointManager, get_checkpoint_manager
from neuroexapt.core.evolution_checkpoint import EvolutionCheckpointManager

# 设置日志 - 简洁格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class GrowableConvBlock(nn.Module):
    """可生长的卷积块"""
    
    def __init__(self, in_channels, out_channels, block_id, stride=1):
        super(GrowableConvBlock, self).__init__()
        
        self.block_id = block_id
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        
        # 基础卷积块
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 并行分支（用于分支生长）
        self.branches = nn.ModuleList()
        
        # 生长历史
        self.growth_history = []
        
        logger.info(f"🧱 Block {block_id} created: {in_channels}→{out_channels}, stride={stride}")
    
    def forward(self, x):
        """前向传播"""
        # 主分支
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        
        # 并行分支（如果有的话）
        if len(self.branches) > 0:
            branch_outputs = []
            for i, branch in enumerate(self.branches):
                try:
                    branch_out = branch(x)
                    
                    # 🔧 修复：安全的形状匹配，避免gradient破坏
                    # 1. 首先处理空间维度匹配
                    if branch_out.shape[2:] != out.shape[2:]:
                        branch_out = F.adaptive_avg_pool2d(branch_out, out.shape[2:])
                    
                    # 2. 处理通道维度匹配 - 使用learnable projection而非零填充
                    if branch_out.shape[1] != out.shape[1]:
                        # 创建或获取通道匹配层
                        if not hasattr(self, f'_channel_adapter_{i}'):
                            # 动态创建通道适配器
                            adapter = nn.Conv2d(
                                branch_out.shape[1], 
                                out.shape[1], 
                                kernel_size=1, 
                                bias=False
                            ).to(branch_out.device)
                            # 使用identity初始化避免破坏已学习特征
                            with torch.no_grad():
                                if branch_out.shape[1] <= out.shape[1]:
                                    # 输入通道少于输出通道：identity + 零初始化
                                    nn.init.zeros_(adapter.weight)
                                    min_channels = min(branch_out.shape[1], out.shape[1])
                                    for c in range(min_channels):
                                        adapter.weight[c, c, 0, 0] = 1.0
                                else:
                                    # 输入通道多于输出通道：取前N个通道
                                    nn.init.zeros_(adapter.weight)
                                    for c in range(out.shape[1]):
                                        adapter.weight[c, c, 0, 0] = 1.0
                            
                            setattr(self, f'_channel_adapter_{i}', adapter)
                        
                        adapter = getattr(self, f'_channel_adapter_{i}')
                        branch_out = adapter(branch_out)
                    
                    branch_outputs.append(branch_out)
                    
                except Exception as e:
                    logger.warning(f"Branch {i} forward failed: {e}")
                    # 🔧 修复：失败时创建安全的零tensor，避免破坏gradient flow
                    safe_output = torch.zeros_like(out)
                    branch_outputs.append(safe_output)
                    continue
            
            # 融合分支输出 - 使用更稳定的融合策略
            if branch_outputs:
                # 使用平均而非求和，避免梯度爆炸
                branch_avg = torch.stack(branch_outputs).mean(dim=0)
                out = out + 0.2 * branch_avg  # 降低分支权重，提高稳定性
        
        return out
    
    def expand_channels(self, new_out_channels):
        """扩展输出通道数"""
        if new_out_channels <= self.out_channels:
            return False
        
        old_channels = self.out_channels
        
        # 获取当前设备
        device = next(self.conv.parameters()).device
        
        # 创建新的卷积层
        new_conv = nn.Conv2d(self.in_channels, new_out_channels, 3, 
                           stride=self.stride, padding=1, bias=False).to(device)
        new_bn = nn.BatchNorm2d(new_out_channels).to(device)
        
        # 参数迁移
        with torch.no_grad():
            # 复制原有参数
            new_conv.weight[:old_channels] = self.conv.weight
            new_bn.weight[:old_channels] = self.bn.weight
            new_bn.bias[:old_channels] = self.bn.bias
            if hasattr(self.bn, 'running_mean'):
                new_bn.running_mean[:old_channels] = self.bn.running_mean
                new_bn.running_var[:old_channels] = self.bn.running_var
        
        # 替换层
        self.conv = new_conv
        self.bn = new_bn
        self.out_channels = new_out_channels
        
        # 🔧 关键修复：精确更新所有分支的输出通道数
        branches_to_remove = []
        for i, branch in enumerate(self.branches):
            try:
                # 清理可能存在的旧通道适配器
                if hasattr(self, f'_channel_adapter_{i}'):
                    delattr(self, f'_channel_adapter_{i}')
                
                # 获取分支的第一个卷积层
                if hasattr(branch, '0') and isinstance(branch[0], nn.Conv2d):
                    old_branch_conv = branch[0]
                    old_branch_bn = branch[1] if len(branch) > 1 and isinstance(branch[1], nn.BatchNorm2d) else None
                    
                    # 创建新的分支卷积层
                    new_branch_conv = nn.Conv2d(
                        old_branch_conv.in_channels, 
                        new_out_channels,  # 使用新的输出通道数
                        old_branch_conv.kernel_size,
                        stride=old_branch_conv.stride,
                        padding=old_branch_conv.padding,
                        dilation=old_branch_conv.dilation,
                        groups=old_branch_conv.groups,
                        bias=old_branch_conv.bias is not None
                    ).to(device)
                    
                    # 创建新的BN层
                    new_branch_bn = nn.BatchNorm2d(new_out_channels).to(device) if old_branch_bn else None
                    
                    # 🔧 安全的参数迁移，避免维度不匹配
                    with torch.no_grad():
                        # 复制卷积权重
                        min_out_channels = min(old_branch_conv.out_channels, new_out_channels)
                        min_in_channels = min(old_branch_conv.in_channels, new_branch_conv.in_channels)
                        
                        # 初始化新权重为零
                        nn.init.zeros_(new_branch_conv.weight)
                        
                        # 复制原有权重到对应位置
                        new_branch_conv.weight[:min_out_channels, :min_in_channels] = \
                            old_branch_conv.weight[:min_out_channels, :min_in_channels]
                        
                        # 如果有bias，也要复制
                        if new_branch_conv.bias is not None and old_branch_conv.bias is not None:
                            new_branch_conv.bias[:min_out_channels] = old_branch_conv.bias[:min_out_channels]
                        
                        # 复制BN参数
                        if new_branch_bn and old_branch_bn:
                            new_branch_bn.weight[:min_out_channels] = old_branch_bn.weight[:min_out_channels]
                            new_branch_bn.bias[:min_out_channels] = old_branch_bn.bias[:min_out_channels]
                            if hasattr(old_branch_bn, 'running_mean') and hasattr(new_branch_bn, 'running_mean'):
                                new_branch_bn.running_mean[:min_out_channels] = old_branch_bn.running_mean[:min_out_channels]
                                new_branch_bn.running_var[:min_out_channels] = old_branch_bn.running_var[:min_out_channels]
                                new_branch_bn.num_batches_tracked.copy_(old_branch_bn.num_batches_tracked)
                    
                    # 重建分支，保持原有结构
                    branch_layers = []
                    branch_layers.append(new_branch_conv)
                    if new_branch_bn:
                        branch_layers.append(new_branch_bn)
                    
                    # 添加激活函数（如果原来有的话）
                    if len(branch) > 2:
                        branch_layers.append(branch[2])
                    elif len(branch) > 1 and not isinstance(branch[1], nn.BatchNorm2d):
                        branch_layers.append(branch[1])
                    else:
                        branch_layers.append(nn.ReLU(inplace=True))
                    
                    self.branches[i] = nn.Sequential(*branch_layers)
                    
                    logger.info(f"🔧 Updated branch {i} output channels: {old_branch_conv.out_channels}→{new_out_channels}")
                    
            except Exception as e:
                logger.warning(f"Failed to update branch {i}: {e}")
                # 标记需要移除的分支
                branches_to_remove.append(i)
        
        # 安全地移除有问题的分支（从后往前移除，避免索引问题）
        for i in reversed(branches_to_remove):
            logger.warning(f"Removing problematic branch {i}")
            self.branches.pop(i)
        
        # 记录生长
        self.growth_history.append({
            'type': 'channel_expansion',
            'old_channels': old_channels,
            'new_channels': new_out_channels,
            'timestamp': time.time()
        })
        
        logger.info(f"🌱 Block {self.block_id} CHANNEL GROWTH: {old_channels}→{new_out_channels} on {device}")
        return True
    
    def add_branch(self):
        """增加并行分支"""
        # 创建新分支
        branch = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, 5, 
                     stride=self.stride, padding=2, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 关键：将新分支移动到正确的设备上
        device = next(self.conv.parameters()).device
        branch = branch.to(device)
        
        self.branches.append(branch)
        
        # 记录生长
        self.growth_history.append({
            'type': 'branch_addition',
            'branch_count': len(self.branches),
            'timestamp': time.time()
        })
        
        logger.info(f"🌿 Block {self.block_id} BRANCH GROWTH: Added branch #{len(self.branches)} on {device}")
        return True

class GrowingNetwork(nn.Module):
    """会真正生长的神经网络"""
    
    def __init__(self, num_classes=10, initial_channels=16, initial_depth=3):
        super(GrowingNetwork, self).__init__()
        
        self.num_classes = num_classes
        self.initial_channels = initial_channels
        self.current_depth = initial_depth
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, initial_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(initial_channels),
            nn.ReLU(inplace=True)
        )
        
        # 动态层列表
        self.layers = nn.ModuleList()
        
        # 构建初始网络（很小！）
        current_channels = initial_channels
        for i in range(initial_depth):
            stride = 2 if i == 1 else 1  # 只在第二层降采样
            out_channels = current_channels * (2 if i == 1 else 1)
            
            block = GrowableConvBlock(current_channels, out_channels, i, stride)
            self.layers.append(block)
            current_channels = out_channels
        
        # 分类头
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(current_channels, num_classes)
        
        # 生长统计
        self.growth_stats = {
            'depth_growths': 0,
            'channel_growths': 0,
            'branch_growths': 0,
            'total_growths': 0,
            'parameter_history': []
        }
        
        # 记录初始参数量
        initial_params = sum(p.numel() for p in self.parameters())
        self.growth_stats['parameter_history'].append({
            'depth': initial_depth,
            'params': initial_params,
            'timestamp': time.time()
        })
        
        logger.info(f"🌱 Growing Network initialized:")
        logger.info(f"   Initial depth: {initial_depth} layers")
        logger.info(f"   Initial channels: {initial_channels}")
        logger.info(f"   Initial parameters: {initial_params:,}")
    
    def forward(self, x):
        """前向传播"""
        x = self.stem(x)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
    
    def grow_depth(self, position=None):
        """增加网络深度 - 真正的层数增长！"""
        if position is None:
            position = len(self.layers) - 1  # 在倒数第二个位置插入
        
        position = max(1, min(position, len(self.layers) - 1))
        
        # 确定新层的通道配置
        if position == 0:
            in_channels = self.initial_channels
            out_channels = self.layers[0].in_channels
        else:
            in_channels = self.layers[position - 1].out_channels
            out_channels = self.layers[position].in_channels
        
        # 创建新层
        new_layer = GrowableConvBlock(in_channels, out_channels, f"grown_{len(self.layers)}", stride=1)
        
        # 移动到正确设备
        if len(self.layers) > 0:
            device = next(self.layers[0].conv.parameters()).device
            new_layer = new_layer.to(device)
        
        # 插入新层
        self.layers.insert(position, new_layer)
        self.current_depth += 1
        
        # 更新统计
        self.growth_stats['depth_growths'] += 1
        self.growth_stats['total_growths'] += 1
        
        # 记录参数变化
        new_params = sum(p.numel() for p in self.parameters())
        self.growth_stats['parameter_history'].append({
            'depth': self.current_depth,
            'params': new_params,
            'timestamp': time.time()
        })
        
        logger.info(f"🌱 DEPTH GROWTH: Added layer at position {position}")
        logger.info(f"   New depth: {self.current_depth} layers")
        logger.info(f"   New parameters: {new_params:,}")
        
        return True
    
    def grow_width(self, layer_idx=None, expansion_factor=1.5):
        """增加网络宽度 - 真正的通道数增长！"""
        if layer_idx is None:
            layer_idx = len(self.layers) // 2  # 选择中间层
        
        if layer_idx >= len(self.layers):
            return False
        
        layer = self.layers[layer_idx]
        new_channels = int(layer.out_channels * expansion_factor)
        
        success = layer.expand_channels(new_channels)
        
        if success:
            # 更新后续层的输入通道数
            self._update_subsequent_layers(layer_idx, new_channels)
            
            # 更新统计
            self.growth_stats['channel_growths'] += 1
            self.growth_stats['total_growths'] += 1
            
            # 记录参数变化
            new_params = sum(p.numel() for p in self.parameters())
            self.growth_stats['parameter_history'].append({
                'depth': self.current_depth,
                'params': new_params,
                'timestamp': time.time()
            })
            
            logger.info(f"🌱 WIDTH GROWTH: Layer {layer_idx} channels expanded")
            logger.info(f"   New parameters: {new_params:,}")
        
        return success
    
    def grow_branches(self, layer_idx=None):
        """增加分支 - 真正的并行处理增长！"""
        if layer_idx is None:
            layer_idx = np.random.randint(0, len(self.layers))
        
        if layer_idx >= len(self.layers):
            return False
        
        layer = self.layers[layer_idx]
        success = layer.add_branch()
        
        if success:
            # 更新统计
            self.growth_stats['branch_growths'] += 1
            self.growth_stats['total_growths'] += 1
            
            # 记录参数变化
            new_params = sum(p.numel() for p in self.parameters())
            self.growth_stats['parameter_history'].append({
                'depth': self.current_depth,
                'params': new_params,
                'timestamp': time.time()
            })
            
            logger.info(f"🌱 BRANCH GROWTH: Layer {layer_idx} added branch")
            logger.info(f"   New parameters: {new_params:,}")
        
        return success
    
    def _update_subsequent_layers(self, start_idx, new_channels):
        """更新后续层的输入通道数"""
        for i in range(start_idx + 1, len(self.layers)):
            layer = self.layers[i]
            
            # 获取设备
            device = next(layer.conv.parameters()).device
            
            # 创建新的卷积层
            new_conv = nn.Conv2d(new_channels, layer.out_channels, 3,
                               stride=layer.stride, padding=1, bias=False).to(device)
            
            # 参数迁移（部分）
            with torch.no_grad():
                min_channels = min(new_channels, layer.in_channels)
                new_conv.weight[:, :min_channels] = layer.conv.weight[:, :min_channels]
            
            layer.conv = new_conv
            layer.in_channels = new_channels
            
            # 🔧 关键修复：安全更新该层所有分支的输入通道数
            branches_to_remove = []
            for j, branch in enumerate(layer.branches):
                try:
                    # 清理可能存在的旧通道适配器
                    if hasattr(layer, f'_channel_adapter_{j}'):
                        delattr(layer, f'_channel_adapter_{j}')
                    
                    # 获取分支的第一个卷积层
                    if hasattr(branch, '0') and isinstance(branch[0], nn.Conv2d):
                        old_branch_conv = branch[0]
                        old_branch_bn = branch[1] if len(branch) > 1 and isinstance(branch[1], nn.BatchNorm2d) else None
                        
                        # 创建新的分支卷积层（更新输入通道数）
                        new_branch_conv = nn.Conv2d(
                            new_channels,  # 使用新的输入通道数
                            old_branch_conv.out_channels,
                            old_branch_conv.kernel_size,
                            stride=old_branch_conv.stride,
                            padding=old_branch_conv.padding,
                            dilation=old_branch_conv.dilation,
                            groups=old_branch_conv.groups,
                            bias=old_branch_conv.bias is not None
                        ).to(device)
                        
                        # 🔧 安全的参数迁移
                        with torch.no_grad():
                            # 计算可以复制的最小通道数
                            min_in_channels = min(new_channels, old_branch_conv.in_channels)
                            min_out_channels = min(old_branch_conv.out_channels, new_branch_conv.out_channels)
                            
                            # 初始化为零
                            nn.init.zeros_(new_branch_conv.weight)
                            
                            # 复制原有权重到对应位置
                            new_branch_conv.weight[:min_out_channels, :min_in_channels] = \
                                old_branch_conv.weight[:min_out_channels, :min_in_channels]
                            
                            # 如果有bias，也要复制
                            if new_branch_conv.bias is not None and old_branch_conv.bias is not None:
                                new_branch_conv.bias[:min_out_channels] = old_branch_conv.bias[:min_out_channels]
                        
                        # 重建分支，保持原有结构
                        branch_layers = []
                        branch_layers.append(new_branch_conv)
                        if old_branch_bn:
                            branch_layers.append(old_branch_bn)  # BN层不需要改变
                        
                        # 添加激活函数
                        if len(branch) > 2:
                            branch_layers.append(branch[2])
                        elif len(branch) > 1 and not isinstance(branch[1], nn.BatchNorm2d):
                            branch_layers.append(branch[1])
                        else:
                            branch_layers.append(nn.ReLU(inplace=True))
                        
                        layer.branches[j] = nn.Sequential(*branch_layers)
                        
                        logger.info(f"🔧 Updated layer {i} branch {j} input channels: {old_branch_conv.in_channels}→{new_channels}")
                        
                except Exception as e:
                    logger.warning(f"Failed to update layer {i} branch {j}: {e}")
                    # 标记需要移除的分支
                    branches_to_remove.append(j)
            
            # 安全地移除有问题的分支（从后往前移除，避免索引问题）
            for j in reversed(branches_to_remove):
                logger.warning(f"Removing problematic branch {j} from layer {i}")
                layer.branches.pop(j)
            
            new_channels = layer.out_channels  # 为下一层准备
    
    def get_architecture_summary(self):
        """获取架构摘要"""
        layer_info = []
        for i, layer in enumerate(self.layers):
            layer_info.append({
                'id': layer.block_id,
                'in_channels': layer.in_channels,
                'out_channels': layer.out_channels,
                'branches': len(layer.branches),
                'growth_history': layer.growth_history
            })
        
        return {
            'depth': self.current_depth,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'growth_stats': self.growth_stats,
            'layers': layer_info
        }

class GrowthController:
    """生长控制器 - 决定何时何地生长"""
    
    def __init__(self):
        self.performance_history = []
        self.growth_decisions = []
        self.last_growth_epoch = -1
        
        # 更激进的生长触发参数 - 冲击95%准确率！
        self.min_epochs_between_growth = 6  # 每6个epoch最少生长1次
        self.improvement_threshold = 0.2  # 更低的改进阈值
        self.forced_growth_interval = 12  # 每12个epoch强制生长一次
        
    def should_grow(self, current_accuracy, epoch):
        """判断是否应该生长 - 更激进的策略"""
        self.performance_history.append(current_accuracy)
        
        # 强制生长：定期必须生长
        if epoch % self.forced_growth_interval == 10:
            logger.info(f"🌱 FORCED GROWTH at epoch {epoch} (scheduled growth)")
            return True
        
        # 如果距离上次生长太久，强制生长
        if epoch - self.last_growth_epoch >= self.min_epochs_between_growth and epoch > 5:
            logger.info(f"🌱 GROWTH TRIGGER at epoch {epoch} (interval-based)")
            return True
        
        # 如果训练早期，积极生长
        if epoch < 20 and epoch % 6 == 5:
            logger.info(f"🌱 EARLY GROWTH at epoch {epoch} (early phase)")
            return True
        
        # 性能停滞检测（更宽松）
        if len(self.performance_history) >= 4:
            recent_performance = self.performance_history[-3:]
            improvement = max(recent_performance) - min(recent_performance)
            
            if improvement < self.improvement_threshold and epoch - self.last_growth_epoch >= 6:
                logger.info(f"🌱 STAGNATION GROWTH at epoch {epoch}")
                logger.info(f"   Recent improvement: {improvement:.2f}%")
                return True
        
        return False
    
    def select_growth_strategy(self, network, current_accuracy):
        """选择生长策略"""
        current_depth = network.current_depth
        total_params = sum(p.numel() for p in network.parameters())
        
        strategies = []
        
        # 基于网络状态和性能选择策略
        if current_accuracy < 30:
            # 低性能：优先增加深度和宽度
            if current_depth < 8:
                strategies.extend(['grow_depth'] * 3)
            strategies.extend(['grow_width'] * 2)
            strategies.append('grow_branches')
            
        elif current_accuracy < 60:
            # 中等性能：平衡发展
            if current_depth < 10:
                strategies.extend(['grow_depth'] * 2)
            strategies.extend(['grow_width'] * 2)
            strategies.extend(['grow_branches'] * 2)
            
        else:
            # 高性能：精细调优
            if current_depth < 12:
                strategies.append('grow_depth')
            strategies.extend(['grow_width', 'grow_branches'] * 2)
        
        # 参数量限制
        if total_params > 500000:  # 50万参数限制
            strategies = [s for s in strategies if s != 'grow_depth']
        
        if not strategies:
            strategies = ['grow_branches']  # 保底策略
        
        selected = np.random.choice(strategies)
        
        logger.info(f"🎯 Selected growth strategy: {selected}")
        logger.info(f"   Current depth: {current_depth}, Parameters: {total_params:,}")
        
        return selected
    
    def execute_growth(self, network, strategy, current_epoch):
        """执行生长策略"""
        success = False
        
        try:
            if strategy == 'grow_depth':
                success = network.grow_depth()
                
            elif strategy == 'grow_width':
                success = network.grow_width(expansion_factor=np.random.uniform(1.3, 1.8))
                
            elif strategy == 'grow_branches':
                success = network.grow_branches()
            
            if success:
                # 更新上次生长时间
                self.last_growth_epoch = current_epoch
                
                decision = {
                    'strategy': strategy,
                    'timestamp': time.time(),
                    'epoch': current_epoch,
                    'depth': network.current_depth,
                    'parameters': sum(p.numel() for p in network.parameters())
                }
                self.growth_decisions.append(decision)
                
                logger.info(f"✅ Growth executed successfully!")
                logger.info(f"   Strategy: {strategy}")
                logger.info(f"   New depth: {network.current_depth}")
                logger.info(f"   New parameters: {decision['parameters']:,}")
            
        except Exception as e:
            logger.error(f"❌ Growth failed: {e}")
            success = False
        
        return success

class GrowingNetworkTrainer:
    """生长网络训练器"""
    
    def __init__(self, experiment_name="growing_network"):
        self.experiment_name = experiment_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 组件
        self.network = None
        self.growth_controller = GrowthController()
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.CrossEntropyLoss()
        
        # 训练状态
        self.best_accuracy = 0.0
        self.training_history = []
        
        # 进化checkpoint管理器
        self.evolution_manager = EvolutionCheckpointManager(experiment_name)
        
        logger.info(f"🌱 Growing Network Trainer initialized")
        logger.info(f"🔧 Device: {self.device}")
        logger.info(f"📚 Evolution checkpoint manager ready")
    
    def setup_data(self, batch_size=128):
        """设置数据"""
        logger.info("📊 Setting up CIFAR-10...")
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        train_dataset = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR10('./data', train=False, transform=transform_test)
        
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        
        logger.info(f"✅ Data ready: {len(train_dataset)} train, {len(test_dataset)} test")
    
    def setup_network(self, initial_channels=16, initial_depth=3):
        """设置网络"""
        self.network = GrowingNetwork(
            num_classes=10,
            initial_channels=initial_channels,
            initial_depth=initial_depth
        ).to(self.device)
        
        self.optimizer = optim.SGD(self.network.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
        
        total_params = sum(p.numel() for p in self.network.parameters())
        logger.info(f"📊 Network setup complete: {total_params:,} parameters")
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.network.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"🚀 Epoch {epoch:02d}")
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.network(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            pbar.set_postfix({
                'Loss': f'{total_loss/(batch_idx+1):.4f}',
                'Acc': f'{100.*correct/total:.2f}%',
                'Depth': self.network.current_depth,
                'Params': f'{sum(p.numel() for p in self.network.parameters()):,}'
            })
        
        return total_loss/len(self.train_loader), 100.*correct/total
    
    def validate(self):
        """验证"""
        self.network.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.network(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        return total_loss/len(self.test_loader), 100.*correct/total
    
    def train(self, epochs=100, initial_channels=16, initial_depth=3, batch_size=128, resume_from=None):
        """主训练流程"""
        logger.info(f"🌱 GROWING NETWORK TRAINING START")
        logger.info(f"📊 Config: epochs={epochs}, initial_channels={initial_channels}, initial_depth={initial_depth}")
        
        start_time = time.time()
        start_epoch = 0
        
        # 设置
        self.setup_data(batch_size)
        self.setup_network(initial_channels, initial_depth)
        
        # 🔄 恢复训练
        if resume_from:
            logger.info(f"🔄 Resuming from checkpoint: {resume_from}")
            try:
                network_state, optimizer_state, scheduler_state, metadata = self.evolution_manager.load_checkpoint(resume_from)
                
                # 恢复网络状态
                self.network.load_state_dict(network_state)
                self.optimizer.load_state_dict(optimizer_state)
                if scheduler_state and self.scheduler:
                    self.scheduler.load_state_dict(scheduler_state)
                
                # 恢复训练统计
                start_epoch = metadata['epoch'] + 1
                self.best_accuracy = metadata['training_stats'].get('best_accuracy', 0.0)
                
                logger.info(f"✅ Successfully resumed from epoch {start_epoch}")
                logger.info(f"   Best accuracy so far: {self.best_accuracy:.2f}%")
                
                # 显示当前架构
                self.display_detailed_architecture()
                
            except Exception as e:
                logger.error(f"❌ Failed to resume from checkpoint: {e}")
                logger.info("🔄 Starting fresh training instead...")
                start_epoch = 0
        
        # 训练循环
        for epoch in range(start_epoch, epochs):
            logger.info(f"\n{'='*80}")
            logger.info(f"Epoch {epoch+1}/{epochs}")
            
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)
            
            # 验证
            test_loss, test_acc = self.validate()
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录统计
            stats = {
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_loss': test_loss,
                'test_accuracy': test_acc,
                'network_summary': self.network.get_architecture_summary()
            }
            self.training_history.append(stats)
            
            # 更新最佳性能（在显示之前）
            if test_acc > self.best_accuracy:
                self.best_accuracy = test_acc
            
            # 输出结果
            arch_summary = self.network.get_architecture_summary()
            logger.info(f"📊 Results:")
            logger.info(f"   Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
            logger.info(f"   Test:  Loss={test_loss:.4f}, Acc={test_acc:.2f}%")
            logger.info(f"   Best:  {self.best_accuracy:.2f}%")
            logger.info(f"   🏗️ Architecture: {arch_summary['depth']} layers, {arch_summary['total_parameters']:,} params")
            
            # 显示生长状态
            epochs_since_growth = epoch - self.growth_controller.last_growth_epoch
            logger.info(f"   🌱 Growth: {arch_summary['growth_stats']['total_growths']} total, {epochs_since_growth} epochs since last")
            
            # 生长决策
            if self.growth_controller.should_grow(test_acc, epoch):
                strategy = self.growth_controller.select_growth_strategy(self.network, test_acc)
                
                # 🔥 关键：生长前保存checkpoint
                logger.info(f"💾 Saving checkpoint before {strategy}...")
                current_stats = {
                    'epoch': epoch,
                    'train_accuracy': train_acc,
                    'test_accuracy': test_acc,
                    'best_accuracy': self.best_accuracy,
                    'train_loss': train_loss,
                    'test_loss': test_loss
                }
                
                pre_growth_checkpoint_id = self.evolution_manager.save_checkpoint(
                    network=self.network,
                    optimizer=self.optimizer, 
                    scheduler=self.scheduler,
                    epoch=epoch,
                    training_stats=current_stats,
                    growth_type=f"pre_{strategy}",
                    parent_id=None  # 当前线性进化
                )
                
                # 保存生长前的状态（用于显示对比）
                pre_growth_params = sum(p.numel() for p in self.network.parameters())
                pre_growth_depth = self.network.current_depth
                
                # 执行生长
                success = self.growth_controller.execute_growth(self.network, strategy, epoch)
                
                if success:
                    # 确保整个网络在正确的设备上
                    self.network = self.network.to(self.device)
                    
                    # 重新创建优化器（参数变了）
                    self.optimizer = optim.SGD(self.network.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
                    self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs-epoch)
                    
                    # 显示生长效果
                    post_growth_params = sum(p.numel() for p in self.network.parameters())
                    post_growth_depth = self.network.current_depth
                    
                    logger.info(f"🎉 NETWORK GROWN SUCCESSFULLY!")
                    logger.info(f"   Depth: {pre_growth_depth} → {post_growth_depth}")
                    logger.info(f"   Parameters: {pre_growth_params:,} → {post_growth_params:,}")
                    logger.info(f"   Parameter increase: +{post_growth_params-pre_growth_params:,}")
                    logger.info(f"   Device check: Network on {next(self.network.parameters()).device}")
            
                    # 💾 生长后也保存checkpoint
                    post_growth_stats = current_stats.copy()
                    post_growth_checkpoint_id = self.evolution_manager.save_checkpoint(
                        network=self.network,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler, 
                        epoch=epoch,
                        training_stats=post_growth_stats,
                        growth_type=strategy,
                        parent_id=pre_growth_checkpoint_id  # 设置父节点关系
                    )
            
            # 定期显示详细架构
            if epoch % 20 == 19:
                self.display_detailed_architecture()
        
        # 训练完成
        total_time = time.time() - start_time
        logger.info(f"\n🎉 GROWING NETWORK TRAINING COMPLETED!")
        logger.info(f"⏱️  Total time: {total_time/60:.1f} minutes")
        logger.info(f"🏆 Best accuracy: {self.best_accuracy:.2f}%")
        
        final_summary = self.network.get_architecture_summary()
        logger.info(f"🌱 Final network:")
        logger.info(f"   Depth: {final_summary['depth']} layers")
        logger.info(f"   Parameters: {final_summary['total_parameters']:,}")
        logger.info(f"   Total growths: {final_summary['growth_stats']['total_growths']}")
        
        # 📊 显示进化树
        self.evolution_manager.display_evolution_tree()
        
        self.display_detailed_architecture()
    
    def display_detailed_architecture(self):
        """显示详细架构信息"""
        summary = self.network.get_architecture_summary()
        
        logger.info(f"\n🏗️ DETAILED ARCHITECTURE:")
        logger.info(f"   Total depth: {summary['depth']} layers")
        logger.info(f"   Total parameters: {summary['total_parameters']:,}")
        logger.info(f"   Growth statistics:")
        for key, value in summary['growth_stats'].items():
            if key != 'parameter_history':
                logger.info(f"     {key}: {value}")
        
        logger.info(f"   Layer details:")
        for i, layer_info in enumerate(summary['layers']):
            branches_info = f", {layer_info['branches']} branches" if layer_info['branches'] > 0 else ""
            logger.info(f"     Layer {i}: {layer_info['in_channels']}→{layer_info['out_channels']}{branches_info}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Growing Network Training')
    parser.add_argument('--epochs', type=int, default=150, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=128, help='批大小')
    parser.add_argument('--initial_channels', type=int, default=32, help='初始通道数')
    parser.add_argument('--initial_depth', type=int, default=4, help='初始深度')
    parser.add_argument('--experiment', type=str, default='growing_network_95', help='实验名称')
    parser.add_argument('--resume_from', type=str, default=None, help='从指定checkpoint恢复训练')
    parser.add_argument('--target_accuracy', type=float, default=95.0, help='目标准确率')
    
    args = parser.parse_args()
    
    logger.info("🌱 GROWING NEURAL NETWORK - REAL STRUCTURAL GROWTH!")
    logger.info(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"📊 Configuration: {vars(args)}")
    
    trainer = GrowingNetworkTrainer(args.experiment)
    
    logger.info(f"🎯 Target accuracy: {args.target_accuracy}%")
    if args.resume_from:
        logger.info(f"🔄 Will resume from: {args.resume_from}")
    
    trainer.train(
        epochs=args.epochs,
        initial_channels=args.initial_channels,
        initial_depth=args.initial_depth,
        batch_size=args.batch_size,
        resume_from=args.resume_from
    )

if __name__ == "__main__":
    main() 