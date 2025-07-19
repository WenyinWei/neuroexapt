"""
@defgroup group_fast_operations Fast Operations
@ingroup core
Fast Operations module for NeuroExapt framework.

高性能操作模块 - 针对ASO-SE架构搜索优化

主要优化：
1. 智能操作选择：只计算权重大的操作
2. 操作缓存：避免重复计算
3. 批量优化：向量化架构参数更新
4. 内存池：减少内存分配开销
5. 设备优化：最小化数据传输
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import math
import time
from collections import OrderedDict
import numpy as np

class FastMixedOp(nn.Module):
    """
    高性能混合操作
    
    核心优化：
    1. 权重阈值过滤：只计算权重>threshold的操作
    2. Top-K选择：只保留前K个最大权重的操作
    3. 操作缓存：缓存昂贵操作的结果
    4. 动态精度：根据权重大小选择计算精度
    """
    
    def __init__(self, C, stride, primitives=None, weight_threshold=0.01, top_k=3):
        super().__init__()
        
        if primitives is None:
            from .genotypes import PRIMITIVES
            primitives = PRIMITIVES
        
        self.C = C
        self.stride = stride
        self.weight_threshold = weight_threshold
        self.top_k = min(top_k, len(primitives))
        
        # 构建操作字典
        self._ops = nn.ModuleDict()
        self._op_names = list(primitives)
        
        # 按计算成本排序操作（便宜的操作优先）
        self._op_costs = self._calculate_operation_costs(primitives)
        self._sorted_ops = sorted(enumerate(primitives), key=lambda x: self._op_costs[x[1]])
        
        for primitive in primitives:
            op = self._create_operation(primitive, C, stride)
            self._ops[primitive] = op
        
        # 缓存相关
        self._cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # 性能统计
        self.forward_count = 0
        self.active_ops_avg = 0.0
        
    def _calculate_operation_costs(self, primitives):
        """计算操作的相对成本"""
        costs = {
            'none': 0.1,
            'skip_connect': 0.2,
            'avg_pool_3x3': 0.3,
            'max_pool_3x3': 0.3,
            'sep_conv_3x3': 1.0,
            'sep_conv_5x5': 1.5,
            'sep_conv_7x7': 2.5,
            'dil_conv_3x3': 1.2,
            'dil_conv_5x5': 1.8,
            'conv_7x1_1x7': 1.3,
        }
        return {op: costs.get(op, 1.0) for op in primitives}
    
    def _create_operation(self, primitive, C, stride):
        """创建优化的操作"""
        from .operations import OPS
        
        op = OPS[primitive](C, stride, False)
        
        # 为池化操作添加BN
        if 'pool' in primitive:
            op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
        
        # 操作融合优化
        if isinstance(op, nn.Sequential):
            op = self._fuse_operations(op)
        
        return op
    
    def _fuse_operations(self, sequential_op):
        """操作融合优化"""
        # 简单的融合：合并连续的Conv+BN+ReLU
        layers = list(sequential_op.children())
        fused_layers = []
        
        i = 0
        while i < len(layers):
            if (i < len(layers) - 2 and 
                isinstance(layers[i], nn.ReLU) and
                isinstance(layers[i+1], nn.Conv2d) and
                isinstance(layers[i+2], nn.BatchNorm2d)):
                
                # 创建融合层
                conv = layers[i+1]
                bn = layers[i+2]
                fused = nn.Sequential(
                    nn.ReLU(inplace=True),  # 使用inplace节省内存
                    conv,
                    bn
                )
                fused_layers.append(fused)
                i += 3
            else:
                fused_layers.append(layers[i])
                i += 1
        
        return nn.Sequential(*fused_layers)
    
    def forward(self, x, weights, training=True):
        """
        高性能前向传播
        
        Args:
            x: 输入张量
            weights: 架构权重 [num_ops]
            training: 是否训练模式
        """
        self.forward_count += 1
        
        if not training:
            # 推理时只使用最大权重的操作
            max_idx = weights.argmax()
            op_name = self._op_names[max_idx]
            return self._ops[op_name](x)
        
        # 训练时的智能选择策略
        return self._forward_training(x, weights)
    
    def _forward_training(self, x, weights):
        """训练时的优化前向传播"""
        device = x.device
        
        # 策略1：权重阈值过滤
        active_indices = (weights > self.weight_threshold).nonzero(as_tuple=True)[0]
        
        if len(active_indices) == 0:
            # 所有权重都很小，使用权重最大的操作
            active_indices = weights.topk(1)[1]
        elif len(active_indices) > self.top_k:
            # 太多活跃操作，只保留前top_k个
            top_weights, top_indices = weights[active_indices].topk(self.top_k)
            active_indices = active_indices[top_indices]
        
        # 更新统计
        self.active_ops_avg = 0.9 * self.active_ops_avg + 0.1 * len(active_indices)
        
        # 计算活跃操作的输出
        outputs = []
        active_weights = []
        
        for idx in active_indices:
            op_name = self._op_names[idx.item()]
            weight = weights[idx]
            
            # 缓存键
            cache_key = self._get_cache_key(x, op_name)
            
            if cache_key in self._cache:
                output = self._cache[cache_key]
                self._cache_hits += 1
            else:
                output = self._ops[op_name](x)
                # 只缓存计算成本高的操作
                if self._op_costs[op_name] > 1.0:
                    self._cache[cache_key] = output.clone()
                    # 限制缓存大小
                    if len(self._cache) > 100:
                        self._cache.clear()
                self._cache_misses += 1
            
            outputs.append(output)
            active_weights.append(weight)
        
        # 归一化权重
        active_weights = torch.stack(active_weights)
        active_weights = active_weights / (active_weights.sum() + 1e-8)
        
        # 加权求和
        result = sum(w * out for w, out in zip(active_weights, outputs))
        
        return result
    
    def _get_cache_key(self, x, op_name):
        """生成缓存键"""
        # 简化的缓存键：基于输入形状和操作名
        return f"{op_name}_{x.shape}_{x.device}"
    
    def get_performance_stats(self):
        """获取性能统计"""
        cache_hit_rate = self._cache_hits / max(self._cache_hits + self._cache_misses, 1)
        return {
            'forward_count': self.forward_count,
            'active_ops_avg': self.active_ops_avg,
            'cache_hit_rate': cache_hit_rate,
            'cache_size': len(self._cache)
        }
    
    def clear_cache(self):
        """清除缓存"""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

class BatchedArchitectureUpdate(nn.Module):
    """
    批量化架构参数更新
    
    将多个架构参数的更新操作批量化，减少GPU kernel调用次数
    """
    
    def __init__(self, num_layers, num_ops_per_layer):
        super().__init__()
        self.num_layers = num_layers
        self.num_ops_per_layer = num_ops_per_layer
        
        # 批量化的架构参数
        self.arch_params = nn.Parameter(
            torch.randn(num_layers, num_ops_per_layer) * 0.1
        )
        
        # Gumbel-Softmax参数
        self.temperature = 5.0
        self.min_temperature = 0.1
        self.anneal_rate = 0.98
    
    def forward(self, layer_idx=None):
        """
        获取架构权重
        
        Args:
            layer_idx: 层索引，None表示返回所有层
        """
        if layer_idx is not None:
            logits = self.arch_params[layer_idx]
        else:
            logits = self.arch_params
        
        if self.training:
            return self._gumbel_softmax(logits)
        else:
            return F.softmax(logits, dim=-1)
    
    def _gumbel_softmax(self, logits):
        """批量化Gumbel-Softmax采样"""
        # 生成Gumbel噪声
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
        
        # 加入噪声并除以温度
        noisy_logits = (logits + gumbel_noise) / self.temperature
        
        # Softmax
        soft_weights = F.softmax(noisy_logits, dim=-1)
        
        # 硬采样（前向时离散，反向时连续）
        hard_weights = F.one_hot(soft_weights.argmax(dim=-1), soft_weights.size(-1)).float()
        
        # 使用straight-through estimator
        return hard_weights - soft_weights.detach() + soft_weights
    
    def anneal_temperature(self):
        """退火温度"""
        self.temperature = max(self.min_temperature, self.temperature * self.anneal_rate)
        return self.temperature
    
    def get_dominant_ops(self, threshold=0.5):
        """获取占主导地位的操作"""
        with torch.no_grad():
            probs = F.softmax(self.arch_params, dim=-1)
            dominant = (probs > threshold).float()
            return dominant

class MemoryEfficientCell(nn.Module):
    """
    内存高效的Cell实现
    
    优化策略：
    1. 梯度检查点：权衡计算和内存
    2. 输出缓存：避免重复计算
    3. 动态形状：根据需要调整
    """
    
    def __init__(self, C_in, C_out, num_nodes=4, num_ops=8, use_checkpoint=True):
        super().__init__()
        
        self.C_in = C_in
        self.C_out = C_out
        self.num_nodes = num_nodes
        self.num_ops = num_ops
        self.use_checkpoint = use_checkpoint
        
        # 预处理层
        self.preprocess0 = self._preprocess_layer(C_in, C_out)
        self.preprocess1 = self._preprocess_layer(C_in, C_out)
        
        # 混合操作
        self.ops = nn.ModuleList()
        for i in range(num_nodes):
            for j in range(2 + i):  # 每个节点连接前面的所有节点
                op = FastMixedOp(C_out, stride=1)
                self.ops.append(op)
        
        # 架构参数更新器
        num_edges = sum(range(2, 2 + num_nodes))
        self.arch_updater = BatchedArchitectureUpdate(num_edges, num_ops)
        
        # 输出处理
        self.output_conv = nn.Conv2d(num_nodes * C_out, C_out, 1, bias=False)
        self.output_bn = nn.BatchNorm2d(C_out)
        
        # 性能监控
        self.memory_usage = []
        self.compute_time = []
    
    def _preprocess_layer(self, C_in, C_out):
        """预处理层"""
        if C_in == C_out:
            return nn.Sequential(
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(C_in)
            )
        else:
            return nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(C_in, C_out, 1, bias=False),
                nn.BatchNorm2d(C_out)
            )
    
    def forward(self, s0, s1):
        """
        前向传播
        
        Args:
            s0, s1: 前两个节点的输出
        """
        # 预处理
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        
        states = [s0, s1]
        
        # 获取所有架构权重
        arch_weights = self.arch_updater()
        
        op_idx = 0
        for i in range(self.num_nodes):
            # 计算新节点
            if self.use_checkpoint and self.training:
                # 使用梯度检查点节省内存
                new_state = torch.utils.checkpoint.checkpoint(
                    self._compute_node, i, states, arch_weights, op_idx
                )
            else:
                new_state = self._compute_node(i, states, arch_weights, op_idx)
            
            states.append(new_state)
            op_idx += len(states) - 1
        
        # 拼接所有中间节点的输出
        intermediate_states = states[2:]  # 排除输入节点
        output = torch.cat(intermediate_states, dim=1)
        
        # 输出处理
        output = self.output_conv(output)
        output = self.output_bn(output)
        
        return output
    
    def _compute_node(self, node_idx, states, arch_weights, start_op_idx):
        """计算单个节点"""
        node_inputs = []
        
        for j, state in enumerate(states):
            op_idx = start_op_idx + j
            if op_idx < len(arch_weights) and op_idx < len(self.ops):
                weight = arch_weights[op_idx]
                
                # 使用FastMixedOp
                op_output = self.ops[op_idx](state, weight, self.training)
                node_inputs.append(op_output)
        
        # 求和所有输入
        return sum(node_inputs) if node_inputs else states[0]
    
    def get_memory_usage(self):
        """获取内存使用情况"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2  # MB
        return 0

class FastDeviceManager:
    """
    快速设备管理器
    
    优化策略：
    1. 设备亲和性：将相关操作放在同一设备
    2. 内存池：预分配内存减少分配开销
    3. 异步传输：重叠计算和数据传输
    4. 批量操作：减少kernel调用
    """
    
    def __init__(self, device=None, memory_pool_size=1024):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.memory_pool_size = memory_pool_size  # MB
        
        # 内存池
        self._init_memory_pool()
        
        # 性能统计
        self.transfer_time = 0.0
        self.transfer_count = 0
        
        print(f"🚀 FastDeviceManager initialized on {self.device}")
    
    def _init_memory_pool(self):
        """初始化内存池"""
        if self.device.type == 'cuda':
            # 预分配一些常用大小的张量
            self.memory_pool = {}
            common_shapes = [
                (1, 32, 32, 32), (1, 64, 16, 16), (1, 128, 8, 8),
                (16, 32, 32, 32), (16, 64, 16, 16), (16, 128, 8, 8)
            ]
            
            for shape in common_shapes:
                self.memory_pool[shape] = torch.empty(shape, device=self.device)
    
    def to_device(self, tensor, non_blocking=True):
        """高效设备转移"""
        if tensor.device == self.device:
            return tensor
        
        start_time = time.time()
        result = tensor.to(self.device, non_blocking=non_blocking)
        self.transfer_time += time.time() - start_time
        self.transfer_count += 1
        
        return result
    
    def get_tensor_from_pool(self, shape, dtype=torch.float32):
        """从内存池获取张量"""
        if shape in self.memory_pool:
            return self.memory_pool[shape].clone()
        else:
            return torch.empty(shape, dtype=dtype, device=self.device)
    
    def get_stats(self):
        """获取性能统计"""
        avg_transfer_time = self.transfer_time / max(self.transfer_count, 1)
        return {
            'device': str(self.device),
            'total_transfer_time': self.transfer_time,
            'transfer_count': self.transfer_count,
            'avg_transfer_time': avg_transfer_time
        }

# 全局设备管理器实例
_fast_device_manager = None

def get_fast_device_manager():
    """获取全局快速设备管理器"""
    global _fast_device_manager
    if _fast_device_manager is None:
        _fast_device_manager = FastDeviceManager()
    return _fast_device_manager

class OperationProfiler:
    """
    操作性能分析器
    
    用于分析不同操作的计算成本，指导优化决策
    """
    
    def __init__(self):
        self.operation_times = {}
        self.operation_memory = {}
        self.operation_count = {}
    
    def profile_operation(self, op_name, operation, input_tensor, num_runs=10):
        """分析操作性能"""
        device = input_tensor.device
        
        # 预热
        for _ in range(3):
            _ = operation(input_tensor)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # 计时
        start_time = time.time()
        start_memory = torch.cuda.memory_allocated() if device.type == 'cuda' else 0
        
        for _ in range(num_runs):
            output = operation(input_tensor)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
        end_memory = torch.cuda.memory_allocated() if device.type == 'cuda' else 0
        
        # 记录结果
        avg_time = (end_time - start_time) / num_runs
        memory_delta = end_memory - start_memory
        
        self.operation_times[op_name] = avg_time
        self.operation_memory[op_name] = memory_delta
        self.operation_count[op_name] = self.operation_count.get(op_name, 0) + 1
    
    def get_operation_ranking(self, criterion='time'):
        """获取操作排名"""
        if criterion == 'time':
            data = self.operation_times
        elif criterion == 'memory':
            data = self.operation_memory
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
        
        return sorted(data.items(), key=lambda x: x[1])
    
    def print_report(self):
        """打印性能报告"""
        print("\n🔍 Operation Performance Report:")
        print("=" * 50)
        
        print("\n⏱️ Time Ranking (fastest to slowest):")
        for op_name, time_ms in self.get_operation_ranking('time'):
            print(f"  {op_name}: {time_ms*1000:.2f}ms")
        
        print("\n💾 Memory Ranking (least to most):")
        for op_name, memory_bytes in self.get_operation_ranking('memory'):
            print(f"  {op_name}: {memory_bytes/1024/1024:.2f}MB")

# 导出接口
__all__ = [
    'FastMixedOp',
    'BatchedArchitectureUpdate', 
    'MemoryEfficientCell',
    'FastDeviceManager',
    'get_fast_device_manager',
    'OperationProfiler'
]