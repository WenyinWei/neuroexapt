"""
defgroup group_fast_math Fast Math
ingroup core
Fast Math module for NeuroExapt framework.
"""

高性能数学计算模块

专门为ASO-SE架构搜索优化的数学运算，包括：
1. 向量化信息论计算
2. 批量化梯度操作
3. 快速相似度计算
4. 内存高效的统计运算
5. GPU优化的数值计算
"""

import torch
import torch.nn.functional as F
import math
import numpy as np
import time
from typing import Tuple, Optional, Union, List
from torch import special

class FastEntropy:
    """高性能熵计算器"""
    
    @staticmethod
    @torch.jit.script
    def entropy_jit(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
        """JIT编译的熵计算"""
        # 使用log_softmax获得数值稳定性
        log_probs = F.log_softmax(x, dim=dim)
        probs = torch.exp(log_probs)
        return -torch.sum(probs * log_probs, dim=dim)
    
    @staticmethod
    def batch_entropy(logits_batch: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """批量熵计算，适用于架构参数"""
        # 形状: [batch_size, num_layers, num_ops] -> [batch_size, num_layers]
        return FastEntropy.entropy_jit(logits_batch, dim=dim)
    
    @staticmethod
    def mutual_information_fast(x: torch.Tensor, y: torch.Tensor, 
                               bins: int = 50) -> torch.Tensor:
        """快速互信息估计"""
        # 使用直方图方法进行快速MI估计
        device = x.device
        
        # 归一化到[0, bins-1]
        x_norm = ((x - x.min()) / (x.max() - x.min() + 1e-8) * (bins - 1)).long()
        y_norm = ((y - y.min()) / (y.max() - y.min() + 1e-8) * (bins - 1)).long()
        
        # 计算联合直方图
        joint_hist = torch.zeros(bins, bins, device=device)
        joint_hist.scatter_add_(0, x_norm.unsqueeze(0).expand(bins, -1), 
                               F.one_hot(y_norm, bins).float().t())
        
        # 归一化为概率
        joint_prob = joint_hist / joint_hist.sum()
        marginal_x = joint_prob.sum(dim=1)
        marginal_y = joint_prob.sum(dim=0)
        
        # 计算MI
        mi = 0.0
        for i in range(bins):
            for j in range(bins):
                if joint_prob[i, j] > 1e-8:
                    mi += joint_prob[i, j] * torch.log(joint_prob[i, j] / 
                                                       (marginal_x[i] * marginal_y[j] + 1e-8))
        
        return mi

class FastGradients:
    """高性能梯度计算"""
    
    @staticmethod
    def compute_gradient_norm_batch(parameters: List[torch.Tensor], 
                                   norm_type: float = 2.0) -> torch.Tensor:
        """批量计算梯度范数"""
        if len(parameters) == 0:
            return torch.tensor(0.0)
        
        device = parameters[0].device
        if norm_type == float('inf'):
            norms = [p.grad.abs().max() for p in parameters if p.grad is not None]
            return torch.stack(norms).max() if norms else torch.tensor(0.0, device=device)
        else:
            squared_norms = [p.grad.norm(dtype=torch.float32) ** norm_type 
                           for p in parameters if p.grad is not None]
            if not squared_norms:
                return torch.tensor(0.0, device=device)
            return torch.stack(squared_norms).sum() ** (1.0 / norm_type)
    
    @staticmethod
    def gradient_similarity(params1: List[torch.Tensor], 
                          params2: List[torch.Tensor]) -> torch.Tensor:
        """计算两组参数梯度的相似性"""
        cos_sims = []
        for p1, p2 in zip(params1, params2):
            if p1.grad is not None and p2.grad is not None:
                cos_sim = F.cosine_similarity(
                    p1.grad.flatten(), p2.grad.flatten(), dim=0
                )
                cos_sims.append(cos_sim)
        
        return torch.stack(cos_sims).mean() if cos_sims else torch.tensor(0.0)
    
    @staticmethod
    def adaptive_gradient_clipping(parameters: List[torch.Tensor], 
                                 max_norm: float = 5.0,
                                 adaptive_factor: float = 0.1) -> float:
        """自适应梯度裁剪"""
        current_norm = FastGradients.compute_gradient_norm_batch(parameters)
        
        if current_norm > max_norm:
            # 动态调整裁剪阈值
            adaptive_max_norm = max_norm + adaptive_factor * (current_norm - max_norm)
            clip_coef = adaptive_max_norm / (current_norm + 1e-8)
            
            for p in parameters:
                if p.grad is not None:
                    p.grad.mul_(clip_coef)
            
            return clip_coef.item()
        
        return 1.0

class FastSimilarity:
    """高性能相似度计算"""
    
    @staticmethod
    @torch.jit.script
    def cosine_similarity_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """JIT优化的余弦相似度矩阵"""
        # 归一化 - JIT需要明确的float类型
        x_norm = F.normalize(x, p=2.0, dim=-1)
        y_norm = F.normalize(y, p=2.0, dim=-1)
        
        # 计算相似度矩阵
        return torch.mm(x_norm, y_norm.t())
    
    @staticmethod
    def architecture_similarity(arch1: torch.Tensor, arch2: torch.Tensor, 
                              method: str = 'cosine') -> torch.Tensor:
        """架构相似度计算"""
        if method == 'cosine':
            return F.cosine_similarity(arch1.flatten(), arch2.flatten(), dim=0)
        elif method == 'kl':
            # KL散度（越小越相似）
            p = F.softmax(arch1.flatten(), dim=0)
            q = F.softmax(arch2.flatten(), dim=0)
            return -torch.sum(p * torch.log(q / p + 1e-8))
        elif method == 'l2':
            return -torch.norm(arch1 - arch2, p=2.0)
        else:
            raise ValueError(f"Unknown similarity method: {method}")
    
    @staticmethod
    def batch_architecture_similarity(arch_batch1: torch.Tensor, 
                                    arch_batch2: torch.Tensor) -> torch.Tensor:
        """批量架构相似度"""
        # 形状: [batch_size, num_layers, num_ops]
        batch_size = arch_batch1.size(0)
        similarities = []
        
        for i in range(batch_size):
            sim = FastSimilarity.architecture_similarity(
                arch_batch1[i], arch_batch2[i]
            )
            similarities.append(sim)
        
        return torch.stack(similarities)

class FastStatistics:
    """高性能统计计算"""
    
    @staticmethod
    @torch.jit.script
    def running_mean_var(x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor, 
                        momentum: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        """运行时均值和方差更新"""
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        
        new_mean = (1 - momentum) * mean + momentum * batch_mean
        new_var = (1 - momentum) * var + momentum * batch_var
        
        return new_mean, new_var
    
    @staticmethod
    def architecture_diversity(arch_batch: torch.Tensor) -> torch.Tensor:
        """计算架构批次的多样性"""
        # 计算批次内架构的平均相似度（越低越多样化）
        batch_size = arch_batch.size(0)
        similarities = []
        
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                sim = FastSimilarity.architecture_similarity(
                    arch_batch[i], arch_batch[j]
                )
                similarities.append(sim)
        
        if similarities:
            return 1.0 - torch.stack(similarities).mean()  # 1 - 平均相似度 = 多样性
        else:
            return torch.tensor(1.0)  # 单个架构认为是完全多样化
    
    @staticmethod
    def confidence_score(weights: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """计算架构权重的置信度"""
        # 使用softmax后的最大值作为置信度
        probs = F.softmax(weights / temperature, dim=-1)
        return probs.max(dim=-1)[0]

class FastNumerical:
    """高性能数值计算"""
    
    @staticmethod
    @torch.jit.script
    def stable_softmax(x: torch.Tensor, dim: int = -1, 
                      temperature: float = 1.0) -> torch.Tensor:
        """数值稳定的softmax"""
        x_scaled = x / temperature
        x_max = x_scaled.max(dim=dim, keepdim=True)[0]
        exp_x = torch.exp(x_scaled - x_max)
        return exp_x / exp_x.sum(dim=dim, keepdim=True)
    
    @staticmethod
    @torch.jit.script
    def gumbel_noise(shape: List[int], device: torch.device) -> torch.Tensor:
        """高效Gumbel噪声生成"""
        u = torch.rand(shape, device=device)
        return -torch.log(-torch.log(u + 1e-8) + 1e-8)
    
    @staticmethod
    def batch_gumbel_softmax(logits: torch.Tensor, temperature: float = 1.0, 
                           hard: bool = True) -> torch.Tensor:
        """批量化Gumbel-Softmax"""
        # 生成Gumbel噪声
        gumbel_noise = FastNumerical.gumbel_noise(
            list(logits.shape), logits.device
        )
        
        # 加入噪声
        noisy_logits = (logits + gumbel_noise) / temperature
        
        # Softmax
        soft_samples = F.softmax(noisy_logits, dim=-1)
        
        if hard:
            # 硬采样
            hard_samples = F.one_hot(
                soft_samples.argmax(dim=-1), soft_samples.size(-1)
            ).float()
            # Straight-through estimator
            return hard_samples - soft_samples.detach() + soft_samples
        
        return soft_samples
    
    @staticmethod
    def differentiable_topk(x: torch.Tensor, k: int, 
                          temperature: float = 1.0) -> torch.Tensor:
        """可微的top-k选择"""
        # 使用Gumbel trick实现可微的top-k
        batch_size, seq_len = x.shape
        
        # 添加Gumbel噪声
        gumbel_noise = FastNumerical.gumbel_noise([batch_size, seq_len], x.device)
        noisy_x = x + gumbel_noise
        
        # 获取top-k indices
        _, topk_indices = torch.topk(noisy_x, k, dim=-1)
        
        # 创建mask
        mask = torch.zeros_like(x)
        mask.scatter_(-1, topk_indices, 1.0)
        
        # 使用softmax进行平滑
        weights = F.softmax(x / temperature, dim=-1)
        
        return mask * weights

class MemoryEfficientOperations:
    """内存高效的操作"""
    
    @staticmethod
    def chunked_matmul(a: torch.Tensor, b: torch.Tensor, 
                      chunk_size: int = 1024) -> torch.Tensor:
        """分块矩阵乘法，节省内存"""
        if a.size(0) <= chunk_size:
            return torch.matmul(a, b)
        
        results = []
        for i in range(0, a.size(0), chunk_size):
            chunk = a[i:i+chunk_size]
            result_chunk = torch.matmul(chunk, b)
            results.append(result_chunk)
        
        return torch.cat(results, dim=0)
    
    @staticmethod
    def memory_efficient_attention(query: torch.Tensor, key: torch.Tensor, 
                                 value: torch.Tensor, chunk_size: int = 512) -> torch.Tensor:
        """内存高效的注意力计算"""
        batch_size, seq_len, d_model = query.shape
        
        if seq_len <= chunk_size:
            # 小序列直接计算
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_model)
            attn_weights = F.softmax(scores, dim=-1)
            return torch.matmul(attn_weights, value)
        
        # 大序列分块计算
        output = torch.zeros_like(query)
        
        for i in range(0, seq_len, chunk_size):
            end_i = min(i + chunk_size, seq_len)
            q_chunk = query[:, i:end_i]
            
            # 计算当前chunk与所有key的attention
            scores = torch.matmul(q_chunk, key.transpose(-2, -1)) / math.sqrt(d_model)
            attn_weights = F.softmax(scores, dim=-1)
            output_chunk = torch.matmul(attn_weights, value)
            
            output[:, i:end_i] = output_chunk
        
        return output

class PerformanceProfiler:
    """性能分析器"""
    
    def __init__(self):
        self.operation_times = {}
        self.memory_usage = {}
        
    def profile_function(self, func_name: str, func, *args, **kwargs):
        """分析函数性能"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start_memory = torch.cuda.memory_allocated()
        
        import time
        start_time = time.perf_counter()
        
        result = func(*args, **kwargs)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            end_memory = torch.cuda.memory_allocated()
            memory_delta = end_memory - start_memory
        else:
            memory_delta = 0
        
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        
        # 记录统计
        if func_name not in self.operation_times:
            self.operation_times[func_name] = []
            self.memory_usage[func_name] = []
        
        self.operation_times[func_name].append(elapsed_time)
        self.memory_usage[func_name].append(memory_delta)
        
        return result
    
    def get_report(self) -> dict:
        """获取性能报告"""
        report = {}
        
        for func_name in self.operation_times:
            times = self.operation_times[func_name]
            memory = self.memory_usage[func_name]
            
            report[func_name] = {
                'avg_time': sum(times) / len(times),
                'total_time': sum(times),
                'call_count': len(times),
                'avg_memory': sum(memory) / len(memory) if memory else 0,
                'peak_memory': max(memory) if memory else 0
            }
        
        return report

# 全局分析器实例
_global_profiler = PerformanceProfiler()

def profile_op(func_name: str):
    """装饰器：自动分析操作性能"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            return _global_profiler.profile_function(func_name, func, *args, **kwargs)
        return wrapper
    return decorator

# 导出接口
__all__ = [
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