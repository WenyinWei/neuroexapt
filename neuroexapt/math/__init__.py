"""
NeuroExapt数学工具模块
包含信息论、优化等数学工具
"""

import torch
import numpy as np

# Note: torch.special.xlogy provides numerically stable x * log(y)
from torch import special

# 基础数学常量
EPS = 1e-8
PI = np.pi

# 信息论相关函数
def entropy(x: torch.Tensor, dim: int = -1, *, already_softmax: bool = False) -> torch.Tensor:
    """计算张量的熵（信息量，Nat）。

    参数:
        x: 输入张量。若 ``already_softmax=True`` 则视为概率分布；否则内部会调用 ``softmax``。
        dim: 取熵的维度。
        already_softmax: 指示 ``x`` 是否已是概率分布，避免重复 ``softmax`` 带来多余开销。
    """
    if not already_softmax:
        x = torch.softmax(x, dim=dim)

    # 使用 xlogy 提供数值稳定性；同时避免 intermediate 张量乘法再 log
    return -torch.sum(special.xlogy(x, x + EPS), dim=dim)

def mutual_information(x: torch.Tensor, y: torch.Tensor, *, already_softmax: bool = False) -> torch.Tensor:
    """简化互信息估计，向量化实现。返回张量而非 Python float 以便梯度反传。"""
    h_x = entropy(x, already_softmax=already_softmax).mean()
    h_y = entropy(y, already_softmax=already_softmax).mean()
    # 拼接前需确保概率归一化；直接用 log(p) trick 不创建额外 softmax
    if not already_softmax:
        xy = torch.softmax(torch.cat([x, y], dim=-1), dim=-1)
    else:
        xy = torch.cat([x, y], dim=-1)
    h_xy = -torch.sum(special.xlogy(xy, xy + EPS), dim=-1).mean()
    return h_x + h_y - h_xy

def kl_divergence(p: torch.Tensor, q: torch.Tensor, dim: int = -1, *, already_softmax: bool = False) -> torch.Tensor:
    """计算 KL 散度 D_\text{KL}(p || q)。返回值为张量，支持梯度。"""
    if not already_softmax:
        p = torch.softmax(p, dim=dim)
        q = torch.softmax(q, dim=dim)
    return torch.sum(special.xlogy(p, (p + EPS) / (q + EPS)), dim=dim)

# 优化相关
def cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """计算余弦相似度"""
    return torch.cosine_similarity(x, y, dim=-1)

def gradient_norm(parameters) -> float:
    """计算梯度范数"""
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

# High-Performance Math Components - NEW!
from .fast_math import (
    FastEntropy, FastGradients, FastSimilarity, FastStatistics,
    FastNumerical, MemoryEfficientOperations, PerformanceProfiler,
    profile_op, _global_profiler
)

# 导出常用函数
__all__ = [
    # 基础数学函数
    'entropy',
    'mutual_information', 
    'kl_divergence',
    'cosine_similarity',
    'gradient_norm',
    'EPS',
    'PI',
    # 高性能数学组件
    'FastEntropy',
    'FastGradients', 
    'FastSimilarity',
    'FastStatistics',
    'FastNumerical',
    'MemoryEfficientOperations',
    'PerformanceProfiler',
    'profile_op',
    '_global_profiler'
]
