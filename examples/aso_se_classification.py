#!/usr/bin/env python3
"""
ASO-SE 神经网络自生长架构系统 - 冲击CIFAR-10 95%准确率

🧬 ASO-SE理论框架 (Alternating Stable Optimization with Stochastic Exploration):
交替式稳定优化与随机探索，解决可微架构搜索的核心矛盾：
- 网络参数和架构参数耦合优化代价巨大
- 解耦优化又会引入破坏性的"架构震荡"

🌱 核心机制：
1. 函数保持突变 - 平滑架构过渡，避免性能剧降
2. Gumbel-Softmax引导探索 - 突破局部最优，智能选择架构
3. 四阶段循环训练 - 稳定优化与探索的完美平衡
4. 渐进式结构生长 - 真正的参数量和深度增长

🎯 目标：CIFAR-10数据集95%+准确率，展示ASO-SE的强大能力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import time
import os
import sys
import json
import math
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Union

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuroexapt.core import CheckpointManager, get_checkpoint_manager
from neuroexapt.core.evolution_checkpoint import EvolutionCheckpointManager
from neuroexapt.core.function_preserving_init import FunctionPreservingInitializer

# 配置简洁日志格式，去除多余前缀
import logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger()

class GumbelSoftmaxSelector:
    """Gumbel-Softmax架构选择器 - 核心探索机制"""
    
    def __init__(self, initial_temp=5.0, min_temp=0.1, anneal_rate=0.98):
        self.initial_temp = initial_temp
        self.min_temp = min_temp
        self.anneal_rate = anneal_rate
        self.current_temp = initial_temp
        
    def sample(self, logits: torch.Tensor, hard=True):
        """使用Gumbel-Softmax进行可微采样"""
        if not self.training:
            # 测试时使用argmax
            return F.one_hot(logits.argmax(dim=-1), logits.size(-1)).float()
        
        # Gumbel噪声
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
        logits_with_noise = (logits + gumbel_noise) / self.current_temp
        
        soft_sample = F.softmax(logits_with_noise, dim=-1)
        
        if hard:
            # 硬采样 - 前向时离散，反向时连续
            hard_sample = F.one_hot(soft_sample.argmax(dim=-1), soft_sample.size(-1)).float()
            return hard_sample - soft_sample.detach() + soft_sample
        
        return soft_sample
    
    def anneal_temperature(self):
        """退火温度"""
        self.current_temp = max(self.min_temp, self.current_temp * self.anneal_rate)
        return self.current_temp

class AdvancedEvolvableBlock(nn.Module):
    """高级可演化块 - 整合所有先进特性"""
    
    def __init__(self, in_channels, out_channels, block_id, stride=1):
        super().__init__()
        
        self.block_id = block_id
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        
        # 多种操作选择 - 架构搜索空间
        self.operations = self._build_operation_space()
        
        # 架构参数 - 用于搜索最优操作组合
        self.alpha_ops = nn.Parameter(torch.randn(len(self.operations)))
        
        # 跳跃连接选择
        self.skip_ops = nn.ModuleList([
            nn.Identity(),  # 直接连接
            nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False) if in_channels != out_channels or stride != 1 else nn.Identity(),  # 1x1投影
        ])
        self.alpha_skip = nn.Parameter(torch.randn(len(self.skip_ops)))
        
        # 并行分支（可动态添加）
        self.branches = nn.ModuleList()
        self.alpha_branches = nn.Parameter(torch.zeros(0))  # 动态大小
        
        # Gumbel-Softmax选择器
        self.gumbel_selector = GumbelSoftmaxSelector()
        
        # 函数保持初始化器
        self.fp_initializer = FunctionPreservingInitializer()
        
        # 演化历史
        self.evolution_history = []
        
        # 性能统计
        self.performance_stats = {
            'forward_count': 0,
            'avg_output_norm': 0.0,
            'gradient_norm': 0.0
        }
        
        print(f"🧱 Block {block_id}: {in_channels}→{out_channels}, stride={stride}, {len(self.operations)} ops")
    
    def _build_operation_space(self):
        """构建丰富的操作搜索空间"""
        ops = nn.ModuleList()
        
        # 1. 标准卷积
        ops.append(nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, 3, 
                     stride=self.stride, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=False)
        ))
        
        # 2. 深度可分离卷积
        if self.in_channels == self.out_channels and self.stride == 1:
            ops.append(nn.Sequential(
                nn.Conv2d(self.in_channels, self.in_channels, 3, 
                         stride=self.stride, padding=1, groups=self.in_channels, bias=False),
                nn.Conv2d(self.in_channels, self.out_channels, 1, bias=False),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU(inplace=False)
            ))
        else:
            # 不能分组时使用1x1卷积
            ops.append(nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, 1, 
                         stride=self.stride, bias=False),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU(inplace=False)
            ))
        
        # 3. 扩张卷积
        ops.append(nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, 3, 
                     stride=self.stride, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=False)
        ))
        
        # 4. 分组卷积
        groups = min(self.in_channels, self.out_channels, 8)
        if self.in_channels % groups == 0 and self.out_channels % groups == 0:
            ops.append(nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, 3, 
                         stride=self.stride, padding=1, groups=groups, bias=False),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU(inplace=False)
            ))
        else:
            # 回退到标准卷积
            ops.append(nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, 3, 
                         stride=self.stride, padding=1, bias=False),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU(inplace=False)
            ))
        
        # 5. 5x5卷积（用两个3x3近似）
        ops.append(nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, 3, 
                     stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 
                     stride=self.stride, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=False)
        ))
        
        return ops
    
    def forward(self, x):
        """前向传播 - ASO-SE架构搜索"""
        # 更新性能统计
        self.performance_stats['forward_count'] += 1
        
        # 使用Gumbel-Softmax选择操作
        op_weights = self.gumbel_selector.sample(self.alpha_ops)
        output = sum(w * op(x) for w, op in zip(op_weights, self.operations))
        
        # 跳跃连接选择
        skip_weights = self.gumbel_selector.sample(self.alpha_skip)
        skip_output = sum(w * op(x) for w, op in zip(skip_weights, self.skip_ops))
        
        # 融合主路径和跳跃连接
        if skip_output.shape == output.shape:
            output = output + 0.3 * skip_output  # 加权融合
        
        # 并行分支（如果有）
        if len(self.branches) > 0 and len(self.alpha_branches) > 0:
            branch_weights = F.softmax(self.alpha_branches, dim=0)
            branch_outputs = []
            
            for branch in self.branches:
                try:
                    branch_out = branch(x)
                    # 形状匹配
                    if branch_out.shape != output.shape:
                        branch_out = self._match_tensor_shape(branch_out, output)
                    branch_outputs.append(branch_out)
                except Exception as e:
                    print(f"Branch error: {e}")
                    branch_outputs.append(torch.zeros_like(output))
            
            if branch_outputs:
                branch_output = sum(w * out for w, out in zip(branch_weights, branch_outputs))
                output = output + 0.2 * branch_output  # 分支贡献权重
        
        # 更新输出统计
        with torch.no_grad():
            self.performance_stats['avg_output_norm'] = 0.9 * self.performance_stats['avg_output_norm'] + 0.1 * output.norm().item()
        
        return output
    
    def _match_tensor_shape(self, source, target):
        """智能张量形状匹配"""
        if source.shape == target.shape:
            return source
        
        # 空间维度匹配
        if source.shape[2:] != target.shape[2:]:
            source = F.adaptive_avg_pool2d(source, target.shape[2:])
        
        # 通道维度匹配
        if source.shape[1] != target.shape[1]:
            if not hasattr(self, '_channel_adapter'):
                self._channel_adapter = nn.Conv2d(
                    source.shape[1], target.shape[1], 1, bias=False
                ).to(source.device)
                
                # 函数保持初始化
                with torch.no_grad():
                    if source.shape[1] <= target.shape[1]:
                        self._channel_adapter.weight.zero_()
                        min_ch = min(source.shape[1], target.shape[1])
                        for i in range(min_ch):
                            self._channel_adapter.weight[i, i, 0, 0] = 1.0
                    else:
                        # 平均池化投影
                        self._channel_adapter.weight.fill_(1.0 / source.shape[1])
            
            source = self._channel_adapter(source)
        
        return source
    
    def grow_branches(self, num_branches=1):
        """增加并行分支 - 真正的结构生长"""
        device = next(self.parameters()).device
        
        for _ in range(num_branches):
            # 创建新分支
            branch = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, 5, 
                         stride=self.stride, padding=2, bias=False),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU(inplace=False)
            ).to(device)
            
            # 函数保持初始化
            self.fp_initializer.initialize_new_branch(branch)
            
            self.branches.append(branch)
        
        # 更新分支权重参数
        new_alpha_branches = torch.zeros(len(self.branches), device=device)
        if len(self.alpha_branches) > 0:
            new_alpha_branches[:len(self.alpha_branches)] = self.alpha_branches
        self.alpha_branches = nn.Parameter(new_alpha_branches)
        
        self.evolution_history.append({
            'type': 'branch_growth',
            'num_branches': num_branches,
            'total_branches': len(self.branches),
            'timestamp': time.time()
        })
        
        print(f"🌿 Block {self.block_id}: Added {num_branches} branches (total: {len(self.branches)})")
        return True
    
    def expand_channels(self, expansion_factor=1.5):
        """扩展通道数 - 真正的参数量增长"""
        new_out_channels = int(self.out_channels * expansion_factor)
        if new_out_channels <= self.out_channels:
            return False
        
        device = next(self.parameters()).device
        old_channels = self.out_channels
        
        # 扩展所有操作的输出通道
        for i, op in enumerate(self.operations):
            new_op = self._expand_operation_channels(op, new_out_channels)
            if new_op is not None:
                self.operations[i] = new_op.to(device)
        
        # 扩展跳跃连接
        for i, skip_op in enumerate(self.skip_ops):
            if isinstance(skip_op, nn.Conv2d):
                new_skip = self._expand_conv_channels(skip_op, new_out_channels)
                if new_skip is not None:
                    self.skip_ops[i] = new_skip.to(device)
        
        # 扩展分支
        for i, branch in enumerate(self.branches):
            new_branch = self._expand_operation_channels(branch, new_out_channels)
            if new_branch is not None:
                self.branches[i] = new_branch.to(device)
        
        self.out_channels = new_out_channels
        
        self.evolution_history.append({
            'type': 'channel_expansion',
            'old_channels': old_channels,
            'new_channels': new_out_channels,
            'expansion_factor': expansion_factor,
            'timestamp': time.time()
        })
        
        print(f"🌱 Block {self.block_id}: Channels {old_channels}→{new_out_channels}")
        return True
    
    def _expand_operation_channels(self, operation, new_out_channels):
        """扩展操作的输出通道数"""
        if isinstance(operation, nn.Sequential):
            new_layers = []
            for layer in operation:
                if isinstance(layer, nn.Conv2d):
                    new_conv = self._expand_conv_channels(layer, new_out_channels)
                    new_layers.append(new_conv if new_conv else layer)
                elif isinstance(layer, nn.BatchNorm2d):
                    new_bn = nn.BatchNorm2d(new_out_channels)
                    # 参数迁移
                    with torch.no_grad():
                        old_channels = layer.num_features
                        min_channels = min(old_channels, new_out_channels)
                        if hasattr(layer, 'weight') and layer.weight is not None:
                            new_bn.weight[:min_channels] = layer.weight[:min_channels]
                        if hasattr(layer, 'bias') and layer.bias is not None:
                            new_bn.bias[:min_channels] = layer.bias[:min_channels]
                        if hasattr(layer, 'running_mean'):
                            new_bn.running_mean[:min_channels] = layer.running_mean[:min_channels]
                        if hasattr(layer, 'running_var'):
                            new_bn.running_var[:min_channels] = layer.running_var[:min_channels]
                    new_layers.append(new_bn)
                else:
                    new_layers.append(layer)
            return nn.Sequential(*new_layers)
        
        return None
    
    def _expand_conv_channels(self, conv_layer, new_out_channels):
        """扩展卷积层的输出通道数"""
        if not isinstance(conv_layer, nn.Conv2d):
            return None
        
        new_conv = nn.Conv2d(
            conv_layer.in_channels,
            new_out_channels,
            conv_layer.kernel_size,
            stride=conv_layer.stride,
            padding=conv_layer.padding,
            dilation=conv_layer.dilation,
            groups=conv_layer.groups if conv_layer.groups == 1 else min(conv_layer.groups, new_out_channels),
            bias=conv_layer.bias is not None
        )
        
        # 函数保持参数迁移
        with torch.no_grad():
            old_out_channels = conv_layer.out_channels
            min_out_channels = min(old_out_channels, new_out_channels)
            
            # 复制权重
            new_conv.weight[:min_out_channels] = conv_layer.weight[:min_out_channels]
            
            # 新增通道用小随机值初始化，避免破坏函数
            if new_out_channels > old_out_channels:
                nn.init.normal_(new_conv.weight[old_out_channels:], mean=0, std=0.01)
            
            # 复制偏置
            if conv_layer.bias is not None:
                new_conv.bias[:min_out_channels] = conv_layer.bias[:min_out_channels]
        
        return new_conv
    
    def get_architecture_weights(self):
        """获取当前架构权重（用于架构参数训练）"""
        return {
            'alpha_ops': self.alpha_ops,
            'alpha_skip': self.alpha_skip,
            'alpha_branches': self.alpha_branches if len(self.alpha_branches) > 0 else None
        }

class ASOSEGrowingNetwork(nn.Module):
    """ASO-SE自生长神经网络 - 完整的四阶段训练框架"""
    
    def __init__(self, num_classes=10, initial_channels=32, initial_depth=4):
        super().__init__()
        
        self.num_classes = num_classes
        self.initial_channels = initial_channels
        self.current_depth = initial_depth
        
        # 输入处理
        self.stem = nn.Sequential(
            nn.Conv2d(3, initial_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(initial_channels),
            nn.ReLU(inplace=True)
        )
        
        # 可演化层
        self.layers = nn.ModuleList()
        self._build_initial_architecture()
        
        # 全局池化和分类器
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.layers[-1].out_channels, num_classes)
        
        # ASO-SE组件
        self.gumbel_selector = GumbelSoftmaxSelector()
        self.fp_initializer = FunctionPreservingInitializer()
        
        # 训练阶段状态
        self.training_phase = "weight_training"  # weight_training, arch_training, mutation, retraining
        self.phase_epoch = 0
        self.cycle_count = 0
        
        # 架构搜索历史
        self.architecture_history = []
        self.performance_history = []
        
        # 生长统计
        self.growth_stats = {
            'depth_growths': 0,
            'channel_growths': 0, 
            'branch_growths': 0,
            'total_growths': 0,
            'parameter_evolution': []
        }
        
        # 记录初始状态
        self._record_current_state("initialization")
        
        print(f"🌱 ASO-SE Network initialized:")
        print(f"   Depth: {self.current_depth}, Channels: {initial_channels}")
        print(f"   Parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def _build_initial_architecture(self):
        """构建初始小网络"""
        current_channels = self.initial_channels
        
        for i in range(self.current_depth):
            # 下采样策略：在深度的1/3和2/3处下采样
            stride = 2 if i in [self.current_depth//3, 2*self.current_depth//3] else 1
            out_channels = current_channels * (2 if stride == 2 else 1)
            
            block = AdvancedEvolvableBlock(
                current_channels, out_channels, f"layer_{i}", stride
            )
            self.layers.append(block)
            current_channels = out_channels
    
    def forward(self, x):
        """前向传播"""
        x = self.stem(x)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
    
    def set_training_phase(self, phase: str):
        """设置训练阶段"""
        valid_phases = ["weight_training", "arch_training", "mutation", "retraining"]
        if phase not in valid_phases:
            raise ValueError(f"Invalid phase: {phase}. Must be one of {valid_phases}")
        
        self.training_phase = phase
        self.phase_epoch = 0
        
        # 配置Gumbel-Softmax
        for layer in self.layers:
            layer.gumbel_selector.training = (phase == "arch_training")
        
        print(f"🔄 Training phase: {phase}")
    
    def get_architecture_parameters(self):
        """获取所有架构参数"""
        arch_params = []
        for layer in self.layers:
            weights = layer.get_architecture_weights()
            arch_params.extend([weights['alpha_ops'], weights['alpha_skip']])
            if weights['alpha_branches'] is not None:
                arch_params.append(weights['alpha_branches'])
        return arch_params
    
    def get_weight_parameters(self):
        """获取所有网络权重参数（非架构参数）"""
        weight_params = []
        arch_param_ids = {id(p) for p in self.get_architecture_parameters()}
        
        for param in self.parameters():
            if id(param) not in arch_param_ids:
                weight_params.append(param)
        
        return weight_params
    
    def grow_depth(self, position=None):
        """增加网络深度 - ASO-SE深度生长"""
        if position is None:
            position = len(self.layers) - 1
        
        position = max(1, min(position, len(self.layers) - 1))
        
        # 确定新层配置
        if position == 0:
            in_channels = self.initial_channels
            out_channels = self.layers[0].in_channels
        else:
            in_channels = self.layers[position-1].out_channels
            out_channels = self.layers[position].in_channels
        
        # 创建新层
        new_layer = AdvancedEvolvableBlock(
            in_channels, out_channels, f"grown_{len(self.layers)}", stride=1
        )
        
        # 函数保持初始化
        self.fp_initializer.initialize_new_layer(new_layer)
        
        # 设备迁移
        device = next(self.parameters()).device
        new_layer = new_layer.to(device)
        
        # 插入层
        self.layers.insert(position, new_layer)
        self.current_depth += 1
        
        # 更新统计
        self.growth_stats['depth_growths'] += 1
        self.growth_stats['total_growths'] += 1
        self._record_current_state("depth_growth")
        
        print(f"🌱 DEPTH GROWTH: Layer added at position {position}")
        print(f"   New depth: {self.current_depth}")
        print(f"   New parameters: {sum(p.numel() for p in self.parameters()):,}")
        
        return True
    
    def grow_width(self, layer_idx=None, expansion_factor=1.4):
        """增加网络宽度 - ASO-SE宽度生长"""
        if layer_idx is None:
            layer_idx = len(self.layers) // 2
        
        if layer_idx >= len(self.layers):
            return False
        
        layer = self.layers[layer_idx]
        success = layer.expand_channels(expansion_factor)
        
        if success:
            # 更新后续层的输入通道
            self._update_subsequent_layers(layer_idx)
            
            # 更新统计
            self.growth_stats['channel_growths'] += 1
            self.growth_stats['total_growths'] += 1
            self._record_current_state("width_growth")
            
            print(f"🌱 WIDTH GROWTH: Layer {layer_idx} expanded by {expansion_factor:.1f}x")
        
        return success
    
    def grow_branches(self, layer_idx=None, num_branches=1):
        """增加分支 - ASO-SE分支生长"""
        if layer_idx is None:
            layer_idx = np.random.randint(0, len(self.layers))
        
        if layer_idx >= len(self.layers):
            return False
        
        layer = self.layers[layer_idx]
        success = layer.grow_branches(num_branches)
        
        if success:
            # 更新统计
            self.growth_stats['branch_growths'] += 1
            self.growth_stats['total_growths'] += 1
            self._record_current_state("branch_growth")
            
            print(f"🌱 BRANCH GROWTH: Layer {layer_idx} added {num_branches} branches")
        
        return success
    
    def _update_subsequent_layers(self, start_idx):
        """更新后续层的输入通道数"""
        if start_idx >= len(self.layers) - 1:
            return
        
        new_channels = self.layers[start_idx].out_channels
        
        for i in range(start_idx + 1, len(self.layers)):
            layer = self.layers[i]
            device = next(layer.parameters()).device
            
            # 更新操作的输入通道
            for j, op in enumerate(layer.operations):
                new_op = self._update_operation_input_channels(op, new_channels)
                if new_op is not None:
                    layer.operations[j] = new_op.to(device)
            
            # 更新跳跃连接
            for j, skip_op in enumerate(layer.skip_ops):
                if isinstance(skip_op, nn.Conv2d):
                    new_skip = self._update_conv_input_channels(skip_op, new_channels)
                    if new_skip is not None:
                        layer.skip_ops[j] = new_skip.to(device)
            
            # 更新分支
            for j, branch in enumerate(layer.branches):
                new_branch = self._update_operation_input_channels(branch, new_channels)
                if new_branch is not None:
                    layer.branches[j] = new_branch.to(device)
            
            layer.in_channels = new_channels
            new_channels = layer.out_channels
        
        # 更新分类器
        final_channels = self.layers[-1].out_channels
        if self.classifier.in_features != final_channels:
            old_classifier = self.classifier
            self.classifier = nn.Linear(final_channels, self.num_classes)
            
            # 参数迁移
            with torch.no_grad():
                min_features = min(old_classifier.in_features, final_channels)
                self.classifier.weight[:, :min_features] = old_classifier.weight[:, :min_features]
                self.classifier.bias.copy_(old_classifier.bias)
            
            device = next(self.parameters()).device
            self.classifier = self.classifier.to(device)
    
    def _update_operation_input_channels(self, operation, new_in_channels):
        """更新操作的输入通道数"""
        if isinstance(operation, nn.Sequential):
            new_layers = []
            for i, layer in enumerate(operation):
                if isinstance(layer, nn.Conv2d) and i == 0:  # 只更新第一个卷积层的输入
                    new_conv = self._update_conv_input_channels(layer, new_in_channels)
                    new_layers.append(new_conv if new_conv else layer)
                else:
                    new_layers.append(layer)
            return nn.Sequential(*new_layers)
        return None
    
    def _update_conv_input_channels(self, conv_layer, new_in_channels):
        """更新卷积层的输入通道数"""
        if not isinstance(conv_layer, nn.Conv2d):
            return None
        
        new_conv = nn.Conv2d(
            new_in_channels,
            conv_layer.out_channels,
            conv_layer.kernel_size,
            stride=conv_layer.stride,
            padding=conv_layer.padding,
            dilation=conv_layer.dilation,
            groups=min(conv_layer.groups, new_in_channels, conv_layer.out_channels),
            bias=conv_layer.bias is not None
        )
        
        # 函数保持参数迁移
        with torch.no_grad():
            old_in_channels = conv_layer.in_channels
            min_in_channels = min(old_in_channels, new_in_channels)
            
            # 复制权重
            new_conv.weight[:, :min_in_channels] = conv_layer.weight[:, :min_in_channels]
            
            # 新增输入通道用小值初始化
            if new_in_channels > old_in_channels:
                nn.init.normal_(new_conv.weight[:, old_in_channels:], mean=0, std=0.01)
            
            # 复制偏置
            if conv_layer.bias is not None:
                new_conv.bias.copy_(conv_layer.bias)
        
        return new_conv
    
    def _record_current_state(self, event_type):
        """记录当前网络状态"""
        state = {
            'event': event_type,
            'timestamp': time.time(),
            'depth': self.current_depth,
            'parameters': sum(p.numel() for p in self.parameters()),
            'growth_stats': self.growth_stats.copy(),
            'training_phase': self.training_phase
        }
        self.growth_stats['parameter_evolution'].append(state)
        
        # 记录架构权重状态（用于分析）
        arch_weights = {}
        for i, layer in enumerate(self.layers):
            weights = layer.get_architecture_weights()
            arch_weights[f'layer_{i}'] = {
                'alpha_ops': weights['alpha_ops'].detach().cpu().numpy().tolist(),
                'alpha_skip': weights['alpha_skip'].detach().cpu().numpy().tolist()
            }
        state['architecture_weights'] = arch_weights
        
        self.architecture_history.append(state)
    
    def anneal_gumbel_temperature(self):
        """退火所有层的Gumbel温度"""
        temps = []
        for layer in self.layers:
            temp = layer.gumbel_selector.anneal_temperature()
            temps.append(temp)
        return sum(temps) / len(temps) if temps else 0
    
    def get_dominant_architecture(self):
        """获取当前占主导地位的架构"""
        arch_description = []
        
        for i, layer in enumerate(self.layers):
            weights = layer.get_architecture_weights()
            
            # 主操作
            dominant_op = weights['alpha_ops'].argmax().item()
            op_prob = F.softmax(weights['alpha_ops'], dim=0)[dominant_op].item()
            
            # 跳跃连接
            dominant_skip = weights['alpha_skip'].argmax().item()
            skip_prob = F.softmax(weights['alpha_skip'], dim=0)[dominant_skip].item()
            
            arch_description.append({
                'layer': i,
                'dominant_op': dominant_op,
                'op_confidence': op_prob,
                'dominant_skip': dominant_skip,
                'skip_confidence': skip_prob,
                'num_branches': len(layer.branches)
            })
        
        return arch_description
    
    def get_architecture_summary(self):
        """获取完整架构摘要"""
        return {
            'depth': self.current_depth,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'growth_stats': self.growth_stats,
            'training_phase': self.training_phase,
            'cycle_count': self.cycle_count,
            'dominant_architecture': self.get_dominant_architecture(),
            'layer_details': [
                {
                    'id': layer.block_id,
                    'in_channels': layer.in_channels,
                    'out_channels': layer.out_channels,
                    'num_operations': len(layer.operations),
                    'num_branches': len(layer.branches),
                    'evolution_history': layer.evolution_history
                } for layer in self.layers
            ]
        }

class ASOSETrainingController:
    """ASO-SE四阶段训练控制器"""
    
    def __init__(self):
        # 四阶段配置
        self.phase_config = {
            'weight_training': {'epochs': 8, 'lr': 0.025},
            'arch_training': {'epochs': 3, 'lr': 3e-4},
            'mutation': {'epochs': 1, 'lr': 0.01},
            'retraining': {'epochs': 6, 'lr': 0.02}
        }
        
        # 生长决策
        self.growth_decisions = []
        self.performance_trend = []
        self.last_growth_cycle = -1
        
        # 生长策略权重（动态调整）
        self.growth_strategy_weights = {
            'grow_depth': 1.0,
            'grow_width': 1.0,
            'grow_branches': 0.8
        }
        
    def should_trigger_growth(self, network, current_cycle, current_accuracy, accuracy_trend):
        """判断是否应该触发生长"""
        # 每3-4个周期必须生长一次
        if current_cycle - self.last_growth_cycle >= 4:
            print(f"🌱 Forced growth trigger (cycle {current_cycle})")
            return True
        
        # 性能停滞检测
        if len(accuracy_trend) >= 3:
            recent_improvement = max(accuracy_trend[-3:]) - min(accuracy_trend[-3:])
            if recent_improvement < 0.5 and current_cycle - self.last_growth_cycle >= 2:
                print(f"🌱 Stagnation growth trigger (improvement: {recent_improvement:.2f}%)")
                return True
        
        # 基于性能阈值的自适应生长
        growth_thresholds = {
            30: 2,   # 30%以下，每2周期生长
            60: 3,   # 30-60%，每3周期生长
            80: 4,   # 60-80%，每4周期生长
            95: 5    # 80%+，每5周期生长
        }
        
        for threshold, interval in growth_thresholds.items():
            if current_accuracy < threshold:
                if current_cycle - self.last_growth_cycle >= interval:
                    print(f"🌱 Performance-based growth trigger (acc: {current_accuracy:.1f}%)")
                    return True
                break
        
        return False
    
    def select_growth_strategy(self, network, current_accuracy, cycle_count):
        """选择生长策略 - 基于性能和网络状态"""
        current_depth = network.current_depth
        total_params = sum(p.numel() for p in network.parameters())
        
        strategies = []
        
        # 基于性能阶段的策略选择
        if current_accuracy < 40:
            # 低性能：积极增加网络容量
            if current_depth < 10:
                strategies.extend(['grow_depth'] * 3)
            strategies.extend(['grow_width'] * 2)
            strategies.append('grow_branches')
            
        elif current_accuracy < 70:
            # 中等性能：平衡发展
            if current_depth < 12:
                strategies.extend(['grow_depth'] * 2)
            strategies.extend(['grow_width'] * 2)
            strategies.extend(['grow_branches'] * 2)
            
        elif current_accuracy < 85:
            # 较高性能：精细调优
            if current_depth < 15:
                strategies.append('grow_depth')
            strategies.extend(['grow_width', 'grow_branches'] * 2)
            
        else:
            # 高性能：分支探索为主
            strategies.extend(['grow_branches'] * 3)
            if current_depth < 18:
                strategies.append('grow_depth')
            strategies.append('grow_width')
        
        # 参数量限制
        if total_params > 800000:  # 80万参数限制
            strategies = [s for s in strategies if s != 'grow_depth']
        if total_params > 1200000:  # 120万参数限制
            strategies = ['grow_branches']
        
        # 应用策略权重
        weighted_strategies = []
        for strategy in strategies:
            weight = self.growth_strategy_weights.get(strategy, 1.0)
            weighted_strategies.extend([strategy] * max(1, int(weight * 2)))
        
        if not weighted_strategies:
            weighted_strategies = ['grow_branches']  # 保底策略
        
        selected = np.random.choice(weighted_strategies)
        
        print(f"🎯 Growth strategy: {selected}")
        print(f"   Network state: depth={current_depth}, params={total_params:,}")
        print(f"   Strategy weights: {self.growth_strategy_weights}")
        
        return selected
    
    def execute_growth(self, network, strategy, cycle_count):
        """执行生长策略"""
        success = False
        growth_details = {}
        
        try:
            pre_growth_params = sum(p.numel() for p in network.parameters())
            pre_growth_depth = network.current_depth
            
            if strategy == 'grow_depth':
                # 智能选择插入位置
                position = self._select_optimal_depth_position(network)
                success = network.grow_depth(position)
                growth_details['position'] = position
                
            elif strategy == 'grow_width':
                # 选择最需要扩展的层
                layer_idx = self._select_optimal_width_layer(network)
                expansion_factor = np.random.uniform(1.3, 1.6)
                success = network.grow_width(layer_idx, expansion_factor)
                growth_details.update({'layer_idx': layer_idx, 'expansion_factor': expansion_factor})
                
            elif strategy == 'grow_branches':
                # 选择合适的层添加分支
                layer_idx = self._select_optimal_branch_layer(network)
                num_branches = np.random.randint(1, 3)
                success = network.grow_branches(layer_idx, num_branches)
                growth_details.update({'layer_idx': layer_idx, 'num_branches': num_branches})
            
            if success:
                self.last_growth_cycle = cycle_count
                
                post_growth_params = sum(p.numel() for p in network.parameters())
                post_growth_depth = network.current_depth
                
                decision = {
                    'strategy': strategy,
                    'cycle': cycle_count,
                    'timestamp': time.time(),
                    'pre_growth': {'depth': pre_growth_depth, 'params': pre_growth_params},
                    'post_growth': {'depth': post_growth_depth, 'params': post_growth_params},
                    'details': growth_details,
                    'param_increase': post_growth_params - pre_growth_params
                }
                self.growth_decisions.append(decision)
                
                # 更新策略权重（成功的策略权重增加）
                self.growth_strategy_weights[strategy] *= 1.1
                
                print(f"✅ Growth executed successfully!")
                print(f"   Depth: {pre_growth_depth} → {post_growth_depth}")
                print(f"   Parameters: {pre_growth_params:,} → {post_growth_params:,}")
                print(f"   Increase: +{post_growth_params - pre_growth_params:,}")
            else:
                # 失败的策略权重降低
                self.growth_strategy_weights[strategy] *= 0.9
                
        except Exception as e:
            print(f"❌ Growth failed: {e}")
            self.growth_strategy_weights.get(strategy, 1.0) * 0.8
            success = False
        
        return success
    
    def _select_optimal_depth_position(self, network):
        """选择最优的深度插入位置"""
        # 在网络后半部分插入，避免破坏早期特征提取
        return max(1, len(network.layers) * 2 // 3)
    
    def _select_optimal_width_layer(self, network):
        """选择最适合宽度扩展的层"""
        # 优先选择中间层，参数效率更高
        return len(network.layers) // 2
    
    def _select_optimal_branch_layer(self, network):
        """选择最适合添加分支的层"""
        # 随机选择，但避免最后一层
        return np.random.randint(0, max(1, len(network.layers) - 1))

class AdvancedDataAugmentation:
    """高级数据增强策略 - 冲击95%准确率"""
    
    @staticmethod
    def get_train_transforms():
        """训练时的强化数据增强"""
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3))
        ])
    
    @staticmethod
    def get_test_transforms():
        """测试时的标准化"""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

class ASOSETrainer:
    """ASO-SE完整训练器 - 四阶段循环训练"""
    
    def __init__(self, experiment_name="aso_se_95"):
        self.experiment_name = experiment_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 核心组件
        self.network = None
        self.training_controller = ASOSETrainingController()
        self.evolution_manager = EvolutionCheckpointManager(experiment_name)
        
        # 优化器（将动态创建）
        self.weight_optimizer = None
        self.arch_optimizer = None
        self.current_optimizer = None
        self.scheduler = None
        
        # 训练状态
        self.current_cycle = 0
        self.current_phase = "weight_training"
        self.phase_epoch = 0
        self.best_accuracy = 0.0
        
        # 历史记录
        self.training_history = []
        self.cycle_results = []
        
        print(f"🌱 ASO-SE Trainer initialized on {self.device}")
        print(f"📊 Target: CIFAR-10 95%+ accuracy")
    
    def setup_data(self, batch_size=128):
        """设置高质量数据加载器"""
        print("📊 Setting up enhanced CIFAR-10 data...")
        
        train_transform = AdvancedDataAugmentation.get_train_transforms()
        test_transform = AdvancedDataAugmentation.get_test_transforms()
        
        train_dataset = torchvision.datasets.CIFAR10(
            './data', train=True, download=True, transform=train_transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            './data', train=False, transform=test_transform
        )
        
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, 
            num_workers=4, pin_memory=True, persistent_workers=True
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True, persistent_workers=True
        )
        
        print(f"✅ Data ready: {len(train_dataset)} train, {len(test_dataset)} test")
        print(f"   Batch size: {batch_size}, Workers: 4")
    
    def setup_network(self, initial_channels=32, initial_depth=4):
        """设置ASO-SE网络"""
        self.network = ASOSEGrowingNetwork(
            num_classes=10,
            initial_channels=initial_channels,
            initial_depth=initial_depth
        ).to(self.device)
        
        self._create_optimizers()
        
        total_params = sum(p.numel() for p in self.network.parameters())
        print(f"📊 ASO-SE Network ready: {total_params:,} parameters")
    
    def _create_optimizers(self):
        """创建专用优化器"""
        # 权重参数优化器
        weight_params = self.network.get_weight_parameters()
        self.weight_optimizer = optim.SGD(
            weight_params, lr=0.025, momentum=0.9, weight_decay=1e-4
        )
        
        # 架构参数优化器
        arch_params = self.network.get_architecture_parameters()
        if arch_params:
            self.arch_optimizer = optim.Adam(arch_params, lr=3e-4, weight_decay=1e-3)
        
        # 当前使用的优化器
        self.current_optimizer = self.weight_optimizer
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.current_optimizer, T_0=50, T_mult=2, eta_min=1e-6
        )
    
    def train_epoch(self, epoch, phase):
        """训练一个epoch"""
        self.network.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # 选择合适的优化器
        if phase == "arch_training":
            optimizer = self.arch_optimizer
            # 冻结权重参数
            for param in self.network.get_weight_parameters():
                param.requires_grad = False
            for param in self.network.get_architecture_parameters():
                param.requires_grad = True
        else:
            optimizer = self.weight_optimizer
            # 训练权重参数
            for param in self.network.get_weight_parameters():
                param.requires_grad = True
            for param in self.network.get_architecture_parameters():
                param.requires_grad = False
        
        criterion = nn.CrossEntropyLoss()
        
        pbar = tqdm(self.train_loader, desc=f"🚀 {phase} Epoch {epoch:02d}")
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.network(data)
            loss = criterion(output, target)
            loss.backward()
            
            # 梯度裁剪
            if phase == "arch_training":
                torch.nn.utils.clip_grad_norm_(self.network.get_architecture_parameters(), 5.0)
            else:
                torch.nn.utils.clip_grad_norm_(self.network.get_weight_parameters(), 5.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            # 实时显示
            if batch_idx % 50 == 0:
                arch_summary = self.network.get_architecture_summary()
                pbar.set_postfix({
                    'Loss': f'{total_loss/(batch_idx+1):.4f}',
                    'Acc': f'{100.*correct/total:.2f}%',
                    'Depth': arch_summary['depth'],
                    'Params': f'{arch_summary["total_parameters"]:,}',
                    'Phase': phase.split('_')[0],
                    'Cycle': self.current_cycle
                })
        
        return total_loss/len(self.train_loader), 100.*correct/total
    
    def validate(self):
        """验证"""
        self.network.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.network(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        return total_loss/len(self.test_loader), 100.*correct/total
    
    def run_training_cycle(self):
        """运行一个完整的ASO-SE四阶段训练周期"""
        cycle_start_time = time.time()
        cycle_results = {}
        
        print(f"\n{'='*80}")
        print(f"🔄 ASO-SE Training Cycle {self.current_cycle + 1}")
        print(f"{'='*80}")
        
        # 阶段1: 权重预热
        print(f"\n🔥 Phase 1: Weight Training (Preheating)")
        self.network.set_training_phase("weight_training")
        weight_results = self._run_phase("weight_training", 8)
        cycle_results['weight_training'] = weight_results
        
        # 阶段2: 架构参数学习
        print(f"\n🧠 Phase 2: Architecture Training (Structure Search)")
        self.network.set_training_phase("arch_training")
        arch_results = self._run_phase("arch_training", 3)
        cycle_results['arch_training'] = arch_results
        
        # 阶段3: 架构突变与稳定
        print(f"\n🧬 Phase 3: Architecture Mutation (Gumbel-Softmax Exploration)")
        mutation_success = self._architecture_mutation()
        cycle_results['mutation_success'] = mutation_success
        
        # 阶段4: 权重再适应
        print(f"\n🔧 Phase 4: Weight Retraining (Adaptation)")
        self.network.set_training_phase("retraining")
        retrain_results = self._run_phase("retraining", 6)
        cycle_results['retraining'] = retrain_results
        
        cycle_time = time.time() - cycle_start_time
        cycle_results['cycle_time'] = cycle_time
        cycle_results['final_accuracy'] = retrain_results['final_test_acc']
        
        self.cycle_results.append(cycle_results)
        
        print(f"\n✅ Cycle {self.current_cycle + 1} completed in {cycle_time/60:.1f} minutes")
        print(f"   Final accuracy: {cycle_results['final_accuracy']:.2f}%")
        print(f"   Best so far: {self.best_accuracy:.2f}%")
        
        return cycle_results
    
    def _run_phase(self, phase_name, num_epochs):
        """运行训练阶段"""
        phase_results = {'epochs': [], 'final_train_acc': 0, 'final_test_acc': 0}
        
        for epoch in range(num_epochs):
            # 训练
            train_loss, train_acc = self.train_epoch(epoch, phase_name)
            
            # 验证
            test_loss, test_acc = self.validate()
            
            # 更新学习率
            if phase_name != "arch_training":
                self.scheduler.step()
            
            # 记录结果
            epoch_result = {
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_loss': test_loss,
                'test_acc': test_acc,
                'lr': self.current_optimizer.param_groups[0]['lr']
            }
            phase_results['epochs'].append(epoch_result)
            
            # 更新最佳性能
            if test_acc > self.best_accuracy:
                self.best_accuracy = test_acc
            
            # Gumbel温度退火
            if phase_name == "arch_training":
                avg_temp = self.network.anneal_gumbel_temperature()
                epoch_result['gumbel_temp'] = avg_temp
            
            print(f"   Epoch {epoch+1}: Train={train_acc:.2f}%, Test={test_acc:.2f}%, Best={self.best_accuracy:.2f}%")
        
        phase_results['final_train_acc'] = phase_results['epochs'][-1]['train_acc']
        phase_results['final_test_acc'] = phase_results['epochs'][-1]['test_acc']
        
        return phase_results
    
    def _architecture_mutation(self):
        """架构突变阶段 - Gumbel-Softmax引导的智能生长"""
        # 分析当前性能趋势
        recent_accuracies = [result['final_accuracy'] for result in self.cycle_results[-3:]]
        if len(recent_accuracies) < 3:
            recent_accuracies = [50.0]  # 默认值
        
        current_accuracy = recent_accuracies[-1] if recent_accuracies else 50.0
        
        # 判断是否需要生长
        should_grow = self.training_controller.should_trigger_growth(
            self.network, self.current_cycle, current_accuracy, recent_accuracies
        )
        
        if should_grow:
            print("🌱 Triggering network growth...")
            
            # 选择生长策略
            strategy = self.training_controller.select_growth_strategy(
                self.network, current_accuracy, self.current_cycle
            )
            
            # 保存生长前状态
            pre_growth_state = self.network.get_architecture_summary()
            
            # 执行生长
            success = self.training_controller.execute_growth(
                self.network, strategy, self.current_cycle
            )
            
            if success:
                # 重新创建优化器（参数可能变化）
                self._create_optimizers()
                
                print("🎉 Network growth successful!")
                print("   Updated optimizers for new parameters")
                
                return True
            else:
                print("❌ Network growth failed")
                return False
        else:
            print("🔄 No growth triggered this cycle")
            
            # 即使不生长，也进行Gumbel-Softmax探索
            print("🎲 Performing Gumbel-Softmax architecture exploration...")
            avg_temp = self.network.anneal_gumbel_temperature()
            print(f"   Current Gumbel temperature: {avg_temp:.3f}")
            
            return False
    
    def train(self, max_cycles=20, initial_channels=32, initial_depth=4, batch_size=128, resume_from=None):
        """主训练流程 - ASO-SE完整四阶段循环"""
        print(f"\n🌱 ASO-SE Training Started")
        print(f"🎯 Target: CIFAR-10 95%+ accuracy")
        print(f"⚙️  Config: max_cycles={max_cycles}, channels={initial_channels}, depth={initial_depth}")
        
        start_time = time.time()
        
        # 设置数据和网络
        self.setup_data(batch_size)
        self.setup_network(initial_channels, initial_depth)
        
        # 恢复训练（如果指定）
        if resume_from:
            print(f"🔄 Resuming from checkpoint: {resume_from}")
            # TODO: 实现恢复逻辑
        
        try:
            # 主训练循环
            for cycle in range(max_cycles):
                self.current_cycle = cycle
                
                # 运行一个完整周期
                cycle_result = self.run_training_cycle()
                
                # 检查是否达到目标
                if cycle_result['final_accuracy'] >= 95.0:
                    print(f"\n🎉 TARGET ACHIEVED! Accuracy: {cycle_result['final_accuracy']:.2f}%")
                    break
                
                # 早停检查
                if self._should_early_stop():
                    print(f"\n⏹️  Early stopping triggered")
                    break
                
                # 显示进度摘要
                self._display_progress_summary()
        
        except KeyboardInterrupt:
            print("\n⏹️  Training interrupted by user")
        
        except Exception as e:
            print(f"\n❌ Training error: {e}")
            raise
        
        finally:
            # 训练完成总结
            total_time = time.time() - start_time
            self._display_final_summary(total_time)
    
    def _should_early_stop(self):
        """早停检查"""
        if len(self.cycle_results) < 5:
            return False
        
        # 检查最近5个周期的改进
        recent_accs = [r['final_accuracy'] for r in self.cycle_results[-5:]]
        improvement = max(recent_accs) - min(recent_accs)
        
        return improvement < 0.5  # 5个周期内改进不到0.5%
    
    def _display_progress_summary(self):
        """显示进度摘要"""
        print(f"\n📊 Progress Summary (Cycle {self.current_cycle + 1}):")
        
        if len(self.cycle_results) >= 3:
            recent_results = self.cycle_results[-3:]
            accs = [r['final_accuracy'] for r in recent_results]
            
            print(f"   Recent accuracies: {accs}")
            print(f"   Trend: {accs[-1] - accs[0]:+.2f}% over 3 cycles")
        
        arch_summary = self.network.get_architecture_summary()
        print(f"   Current network: {arch_summary['depth']} layers, {arch_summary['total_parameters']:,} params")
        print(f"   Growth stats: {arch_summary['growth_stats']}")
        
        # 显示占主导地位的架构
        dominant_arch = self.network.get_dominant_architecture()
        print(f"   Dominant operations: {[layer['dominant_op'] for layer in dominant_arch[:5]]}")
    
    def _display_final_summary(self, total_time):
        """显示最终总结"""
        print(f"\n{'='*80}")
        print(f"🎉 ASO-SE Training Completed!")
        print(f"{'='*80}")
        
        print(f"⏱️  Total time: {total_time/3600:.1f} hours ({total_time/60:.1f} minutes)")
        print(f"🔄 Total cycles: {len(self.cycle_results)}")
        print(f"🏆 Best accuracy: {self.best_accuracy:.2f}%")
        
        if self.cycle_results:
            final_result = self.cycle_results[-1]
            print(f"📊 Final accuracy: {final_result['final_accuracy']:.2f}%")
        
        arch_summary = self.network.get_architecture_summary()
        print(f"🏗️  Final architecture:")
        print(f"   Depth: {arch_summary['depth']} layers")
        print(f"   Parameters: {arch_summary['total_parameters']:,}")
        print(f"   Total growths: {arch_summary['growth_stats']['total_growths']}")
        
        print(f"\n🧬 Growth breakdown:")
        growth_stats = arch_summary['growth_stats']
        for growth_type in ['depth_growths', 'channel_growths', 'branch_growths']:
            print(f"   {growth_type}: {growth_stats[growth_type]}")
        
        # 显示最终占主导地位的架构
        print(f"\n🎯 Final dominant architecture:")
        dominant_arch = self.network.get_dominant_architecture()
        for i, layer in enumerate(dominant_arch[:8]):  # 显示前8层
            print(f"   Layer {i}: Op{layer['dominant_op']}({layer['op_confidence']:.2f}), Skip{layer['dominant_skip']}({layer['skip_confidence']:.2f}), Branches{layer['num_branches']}")
        
        # 保存最终模型
        final_checkpoint = self.evolution_manager.save_checkpoint(
            network=self.network,
            optimizer=self.weight_optimizer,
            scheduler=self.scheduler,
            epoch=self.current_cycle,
            training_stats={'best_accuracy': self.best_accuracy},
            growth_type="final_model"
        )
        print(f"💾 Final model saved: {final_checkpoint}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ASO-SE Neural Network Training')
    parser.add_argument('--cycles', type=int, default=25, help='Maximum training cycles')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--initial_channels', type=int, default=32, help='Initial channels')
    parser.add_argument('--initial_depth', type=int, default=4, help='Initial depth')
    parser.add_argument('--experiment', type=str, default='aso_se_95', help='Experiment name')
    parser.add_argument('--resume_from', type=str, default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    print("🧬 ASO-SE: Alternating Stable Optimization with Stochastic Exploration")
    print("🎯 Target: CIFAR-10 95%+ Accuracy with True Neural Architecture Growth")
    print(f"⏰ Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📋 Config: {vars(args)}")
    
    # 创建训练器
    trainer = ASOSETrainer(args.experiment)
    
    # 开始训练
    trainer.train(
        max_cycles=args.cycles,
        initial_channels=args.initial_channels,
        initial_depth=args.initial_depth,
        batch_size=args.batch_size,
        resume_from=args.resume_from
    )

if __name__ == "__main__":
    main() 