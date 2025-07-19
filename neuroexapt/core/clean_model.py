#!/usr/bin/env python3
"""
@defgroup group_clean_model Clean Model
@ingroup core
Clean Model module for NeuroExapt framework.


🔧 完全重构的干净模型实现
修复所有通道数计算问题，提供可靠的基础架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import time

from .operations import OPS, MixedOp, ReLUConvBN, FactorizedReduce
from .genotypes import PRIMITIVES, Genotype

class CleanCell(nn.Module):
    """
    完全重构的Cell实现 - 保证通道数计算正确
    """
    def __init__(self, steps: int, block_multiplier: int, 
                 C_prev_prev: int, C_prev: int, C: int, 
                 reduction: bool, reduction_prev: bool):
        super(CleanCell, self).__init__()
        
        self.reduction = reduction
        self.steps = steps
        self.block_multiplier = block_multiplier
        
        # 🔧 关键修复：确保预处理层通道数正确
        if reduction_prev:
            # 前一层是reduction，需要FactorizedReduce
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            # 前一层是normal，使用1x1 conv调整通道数
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        
        # s1的预处理 - 从C_prev调整到C
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        
        # 创建MixedOp - 所有操作的输入输出通道数都是C
        self._ops = nn.ModuleList()
        for i in range(steps):
            for j in range(2 + i):  # 每个节点连接到前面所有节点
                # 在reduction cell中，前两个节点的操作使用stride=2
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)  # 使用最基础的MixedOp
                self._ops.append(op)
    
    def forward(self, s0: torch.Tensor, s1: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """前向传播 - 确保尺寸和通道数正确"""
        
        # 预处理：调整输入到统一的通道数C
        s0 = self.preprocess0(s0)  # [B, C_prev_prev, H, W] -> [B, C, H', W']
        s1 = self.preprocess1(s1)  # [B, C_prev, H, W] -> [B, C, H, W]
        
        # 确保s0和s1的空间尺寸匹配
        if s0.shape[2:] != s1.shape[2:]:
            # 如果尺寸不匹配，调整s0到s1的尺寸
            s0 = F.interpolate(s0, size=s1.shape[2:], mode='bilinear', align_corners=False)
        
        states = [s0, s1]  # 初始状态
        offset = 0
        
        # 逐步构建中间节点
        for i in range(self.steps):
            # 收集当前节点的所有输入
            node_inputs = []
            for j in range(len(states)):  # 连接到前面所有状态
                op_idx = offset + j
                if op_idx < len(self._ops) and op_idx < len(weights):
                    h = self._ops[op_idx](states[j], weights[op_idx])
                    node_inputs.append(h)
            
            # 求和得到新节点
            if node_inputs:
                # 确保所有输入尺寸匹配
                target_size = node_inputs[0].shape[2:]
                aligned_inputs = []
                for inp in node_inputs:
                    if inp.shape[2:] != target_size:
                        inp = F.interpolate(inp, size=target_size, mode='bilinear', align_corners=False)
                    aligned_inputs.append(inp)
                
                new_state = sum(aligned_inputs)
            else:
                # 如果没有有效输入，创建零张量
                new_state = torch.zeros_like(states[-1])
            
            states.append(new_state)
            offset += len(states) - 1  # 更新offset
        
        # 输出：连接最后block_multiplier个状态
        output_states = states[-self.block_multiplier:]
        
        # 确保所有输出状态尺寸匹配
        if len(output_states) > 1:
            target_size = output_states[0].shape[2:]
            aligned_outputs = []
            for state in output_states:
                if state.shape[2:] != target_size:
                    state = F.interpolate(state, size=target_size, mode='bilinear', align_corners=False)
                aligned_outputs.append(state)
            output_states = aligned_outputs
        
        result = torch.cat(output_states, dim=1)  # [B, C*block_multiplier, H, W]
        return result

class CleanNetwork(nn.Module):
    """
    完全重构的Network实现 - 保证通道数流动正确
    """
    def __init__(self, C: int, num_classes: int, layers: int, 
                 steps: int = 4, block_multiplier: int = 4):
        super(CleanNetwork, self).__init__()
        
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._steps = steps
        self._block_multiplier = block_multiplier
        
        # Stem: 3 -> C*block_multiplier
        stem_channels = C * block_multiplier
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels)
        )
        
        # 🔧 关键修复：正确的通道数初始化和流动
        self.cells = nn.ModuleList()
        
        # 初始通道设置
        C_prev_prev = stem_channels  # stem输出
        C_prev = stem_channels       # 第一个cell的输入
        C_curr = C                   # 每个cell内部的通道数
        reduction_prev = False
        
        print(f"🔧 干净模型通道数流动:")
        print(f"   Stem: 3 -> {stem_channels}")
        print(f"   初始: C_prev_prev={C_prev_prev}, C_prev={C_prev}, C_curr={C_curr}")
        
        for i in range(layers):
            # 确定是否是reduction layer
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2  # reduction layer通道数翻倍
                reduction = True
                print(f"   Layer {i}: Reduction层, C_curr={C_curr}")
            else:
                reduction = False
                print(f"   Layer {i}: Normal层, C_curr={C_curr}")
            
            # 创建Cell
            cell = CleanCell(
                steps=steps,
                block_multiplier=block_multiplier,
                C_prev_prev=C_prev_prev,
                C_prev=C_prev, 
                C=C_curr,
                reduction=reduction,
                reduction_prev=reduction_prev
            )
            self.cells.append(cell)
            
            # 更新下一轮的通道数
            reduction_prev = reduction
            C_prev_prev = C_prev
            C_prev = C_curr * block_multiplier  # cell输出的通道数
            
            print(f"     -> 输出: {C_prev}, 下一轮: C_prev_prev={C_prev_prev}, C_prev={C_prev}")
        
        # 分类器
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)
        
        # 初始化架构参数
        self._initialize_alphas()
        
        print(f"   最终分类器输入: {C_prev} -> {num_classes}")
        print(f"✅ 干净模型构建完成!")
    
    def _initialize_alphas(self):
        """初始化架构参数"""
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)
        
        self.alphas_normal = nn.Parameter(1e-3 * torch.randn(k, num_ops))
        self.alphas_reduce = nn.Parameter(1e-3 * torch.randn(k, num_ops))
        
        self._arch_parameters = [self.alphas_normal, self.alphas_reduce]
    
    def arch_parameters(self):
        return self._arch_parameters
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """前向传播 - 确保尺寸和通道数正确"""
        
        # 预计算架构权重
        weights_normal = F.softmax(self.alphas_normal, dim=-1)
        weights_reduce = F.softmax(self.alphas_reduce, dim=-1)
        
        # Stem处理
        s0 = s1 = self.stem(input)  # [B, 3, 32, 32] -> [B, C*4, 32, 32]
        
        # 逐层处理
        for i, cell in enumerate(self.cells):
            # 选择权重
            if cell.reduction:
                weights = weights_reduce
            else:
                weights = weights_normal
            
            # Cell前向传播
            s0, s1 = s1, cell(s0, s1, weights)
        
        # 全局池化和分类
        out = self.global_pooling(s1)  # [B, C_final, H, W] -> [B, C_final, 1, 1]
        logits = self.classifier(out.view(out.size(0), -1))  # [B, C_final] -> [B, num_classes]
        
        return logits
    
    def genotype(self):
        """解码架构"""
        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) 
                              if k != PRIMITIVES.index('none')))[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    # 确保k_best不为None
                    if k_best is None:
                        k_best = 1  # 默认使用skip_connect
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene
        
        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())
        
        concat = range(2 + self._steps - self._block_multiplier, self._steps + 2)
        return Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )

def create_clean_network(C: int = 16, num_classes: int = 10, layers: int = 8) -> CleanNetwork:
    """
    创建干净的网络实例
    
    Args:
        C: 基础通道数
        num_classes: 分类数
        layers: 层数
    
    Returns:
        干净的网络模型
    """
    return CleanNetwork(C=C, num_classes=num_classes, layers=layers) 