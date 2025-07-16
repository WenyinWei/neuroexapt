#!/usr/bin/env python3
"""
高效操作模块

实现参数高效的自适应架构搜索策略：
1. 参数共享MixedOp - 相同操作跨层共享参数
2. 动态操作剪枝 - 实时剪除低权重操作  
3. 轻量级候选集 - 减少候选操作数量
4. 渐进式搜索 - 从简单到复杂的搜索策略
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math

from .operations import OPS, SepConv, DilConv, Identity, Zero
from .genotypes import PRIMITIVES

# 轻量级候选操作集（减少参数量）
EFFICIENT_PRIMITIVES = [
    'none',
    'skip_connect', 
    'sep_conv_3x3',
    'sep_conv_5x5',
    'avg_pool_3x3',
    'max_pool_3x3'
]

class SharedOperationPool(nn.Module):
    """
    共享操作池 - 所有MixedOp共享相同的操作实例
    
    这样可以大幅减少参数量，因为相同类型的操作在不同位置共享参数
    """
    
    def __init__(self, C: int, stride: int = 1):
        super().__init__()
        self.C = C
        self.stride = stride
        
        # 创建共享的操作实例
        self.shared_ops = nn.ModuleDict()
        
        for primitive in EFFICIENT_PRIMITIVES:
            if primitive == 'none':
                self.shared_ops[primitive] = Zero(stride)
            elif primitive == 'skip_connect':
                if stride == 1:
                    self.shared_ops[primitive] = Identity()
                else:
                    self.shared_ops[primitive] = nn.Sequential(
                        nn.AvgPool2d(1, stride=stride, padding=0),
                        nn.Conv2d(C, C, 1, stride=1, padding=0, bias=False),
                        nn.BatchNorm2d(C, affine=False)
                    )
            else:
                op = OPS[primitive](C, stride, False)
                if 'pool' in primitive:
                    op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
                self.shared_ops[primitive] = op
    
    def forward(self, x: torch.Tensor, operation: str) -> torch.Tensor:
        """执行指定操作"""
        return self.shared_ops[operation](x)

class EfficientMixedOp(nn.Module):
    """
    高效混合操作
    
    特点：
    1. 使用共享操作池减少参数
    2. 支持动态剪枝
    3. 可配置的候选操作集
    """
    
    def __init__(self, C: int, stride: int, operation_pool: SharedOperationPool, 
                 enable_pruning: bool = True, pruning_threshold: float = 0.01):
        super().__init__()
        self.C = C
        self.stride = stride
        self.operation_pool = operation_pool
        self.enable_pruning = enable_pruning
        self.pruning_threshold = pruning_threshold
        
        # 操作候选列表
        self.primitives = EFFICIENT_PRIMITIVES
        self.num_ops = len(self.primitives)
        
        # 活跃操作掩码（用于动态剪枝）
        self.register_buffer('active_mask', torch.ones(self.num_ops, dtype=torch.bool))
        
        # 操作使用统计（用于分析）
        self.register_buffer('op_usage_count', torch.zeros(self.num_ops))
    
    def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [B, C, H, W]
            weights: 操作权重 [num_ops]
        """
        # 动态剪枝：只计算权重大于阈值的操作
        if self.enable_pruning and self.training:
            active_indices = torch.where(weights > self.pruning_threshold)[0]
            if len(active_indices) == 0:
                # 如果所有权重都太小，保留权重最大的操作
                active_indices = torch.argmax(weights).unsqueeze(0)
        else:
            active_indices = torch.arange(self.num_ops, device=weights.device)
        
        # 计算活跃操作的输出
        outputs = []
        active_weights = []
        
        for i in active_indices:
            primitive = self.primitives[i]
            op_output = self.operation_pool(x, primitive)
            outputs.append(op_output)
            active_weights.append(weights[i])
            
            # 更新使用统计
            self.op_usage_count[i] += 1
        
        if len(outputs) == 1:
            return outputs[0] * active_weights[0]
        else:
            # 归一化权重
            active_weights = torch.stack(active_weights)
            active_weights = F.softmax(active_weights, dim=0)
            
            # 加权求和
            result = outputs[0] * active_weights[0]
            for i in range(1, len(outputs)):
                result = result + outputs[i] * active_weights[i]
            
            return result
    
    def get_active_operations(self, weights: torch.Tensor) -> List[str]:
        """获取当前活跃的操作列表"""
        active_indices = torch.where(weights > self.pruning_threshold)[0]
        return [self.primitives[i] for i in active_indices]
    
    def get_operation_stats(self) -> Dict[str, int]:
        """获取操作使用统计"""
        stats = {}
        for i, primitive in enumerate(self.primitives):
            stats[primitive] = self.op_usage_count[i].item()
        return stats

class EfficientCell(nn.Module):
    """
    高效Cell实现
    
    使用参数共享和动态剪枝来减少计算开销
    """
    
    def __init__(self, steps: int, block_multiplier: int, C_prev_prev: int, 
                 C_prev: int, C: int, reduction: bool, reduction_prev: bool,
                 shared_normal_pool: SharedOperationPool, shared_reduce_pool: SharedOperationPool):
        super().__init__()
        self.reduction = reduction
        self.steps = steps
        self.block_multiplier = block_multiplier
        
        # 预处理层（仍需要独立参数）
        if reduction_prev:
            self.preprocess0 = nn.Sequential(
                nn.AvgPool2d(1, stride=2, padding=0),
                nn.Conv2d(C_prev_prev, C, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(C, affine=False)
            )
        else:
            self.preprocess0 = nn.Sequential(
                nn.Conv2d(C_prev_prev, C, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(C, affine=False)
            )
        
        self.preprocess1 = nn.Sequential(
            nn.Conv2d(C_prev, C, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(C, affine=False)
        )
        
        # 高效MixedOp（使用共享操作池）
        self._ops = nn.ModuleList()
        for i in range(self.steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                if reduction:
                    op = EfficientMixedOp(C, stride, shared_reduce_pool)
                else:
                    op = EfficientMixedOp(C, stride, shared_normal_pool)
                self._ops.append(op)
    
    def forward(self, s0: torch.Tensor, s1: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        
        states = [s0, s1]
        offset = 0
        
        for i in range(self.steps):
            s = 0
            for j in range(2 + i):
                op = self._ops[offset + j]
                h = op(states[j], weights[offset + j])
                s = s + h
            
            offset += len(states)
            states.append(s)
        
        return torch.cat(states[-self.block_multiplier:], dim=1)

class ProgressiveArchitectureSearch:
    """
    渐进式架构搜索
    
    从简单架构开始，逐步增加复杂度：
    1. 第1阶段：只使用基础操作（skip, pool）
    2. 第2阶段：添加3x3卷积
    3. 第3阶段：添加5x5卷积和dilated卷积
    """
    
    def __init__(self):
        self.stage = 1
        self.stage_epochs = [5, 10, 15]  # 每个阶段的epoch数
        self.current_epoch = 0
        
        # 各阶段的操作集
        self.stage_primitives = {
            1: ['none', 'skip_connect', 'avg_pool_3x3', 'max_pool_3x3'],
            2: ['none', 'skip_connect', 'avg_pool_3x3', 'max_pool_3x3', 'sep_conv_3x3'],
            3: EFFICIENT_PRIMITIVES
        }
    
    def update_epoch(self, epoch: int):
        """更新当前epoch，自动切换搜索阶段"""
        self.current_epoch = epoch
        
        if epoch < self.stage_epochs[0]:
            self.stage = 1
        elif epoch < self.stage_epochs[1]:
            self.stage = 2
        else:
            self.stage = 3
    
    def get_current_primitives(self) -> List[str]:
        """获取当前阶段的操作集"""
        return self.stage_primitives[self.stage]
    
    def should_expand_search(self) -> bool:
        """是否应该扩展搜索空间"""
        return self.current_epoch in self.stage_epochs

def create_efficient_network(C: int, num_classes: int, layers: int, 
                           use_progressive_search: bool = True) -> Tuple[nn.Module, Dict]:
    """
    创建参数高效的自适应网络
    
    Returns:
        model: 网络模型
        optimization_info: 优化信息
    """
    # 创建共享操作池
    shared_normal_pool = SharedOperationPool(C, stride=1)
    shared_reduce_pool = SharedOperationPool(C, stride=2)
    
    # 计算参数节省
    traditional_params = layers * 4 * 2 * len(PRIMITIVES) * (C * C * 9)  # 粗略估算
    efficient_params = len(EFFICIENT_PRIMITIVES) * (C * C * 9) * 2  # 共享池参数
    param_reduction = (traditional_params - efficient_params) / traditional_params
    
    optimization_info = {
        'parameter_reduction': param_reduction,
        'shared_operations': len(EFFICIENT_PRIMITIVES),
        'traditional_operations': layers * 4 * 2 * len(PRIMITIVES),
        'efficiency_ratio': (layers * 4 * 2 * len(PRIMITIVES)) / len(EFFICIENT_PRIMITIVES)
    }
    
    print(f"💡 参数效率优化:")
    print(f"   传统方法操作数: {optimization_info['traditional_operations']}")
    print(f"   高效方法操作数: {optimization_info['shared_operations']}")
    print(f"   参数减少估算: {param_reduction*100:.1f}%")
    print(f"   效率提升比: {optimization_info['efficiency_ratio']:.1f}x")
    
    return None, optimization_info  # 暂时返回None，稍后实现完整网络

if __name__ == "__main__":
    # 测试参数效率
    info = create_efficient_network(16, 10, 6)
    print(f"优化信息: {info}") 