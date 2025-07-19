
"""
\defgroup group_model Model
\ingroup core
Model module for NeuroExapt framework.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import psutil
import gc
from typing import Dict, List, Optional
from .operations import OPS, FactorizedReduce, ReLUConvBN, MixedOp
from .genotypes import PRIMITIVES, Genotype

class PerformanceMonitor:
    """性能监控工具类"""
    def __init__(self):
        self.times: Dict[str, List[float]] = {}
        self.memory_usage: List[float] = []
        self.start_time: Optional[float] = None
        
    def start_timer(self, name: str):
        """开始计时"""
        self.start_time = time.perf_counter()
        
    def end_timer(self, name: str):
        """结束计时并记录"""
        if self.start_time is not None:
            elapsed = time.perf_counter() - self.start_time
            if name not in self.times:
                self.times[name] = []
            self.times[name].append(elapsed)
            self.start_time = None
            return elapsed
        return 0.0
    
    def log_memory(self):
        """记录GPU内存使用"""
        if torch.cuda.is_available():
            memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
            self.memory_usage.append(memory_mb)
            return memory_mb
        return 0.0
    
    def get_stats(self) -> Dict[str, float]:
        """获取统计信息"""
        stats = {}
        for name, times in self.times.items():
            if times:
                stats[f"{name}_avg"] = sum(times) / len(times)
                stats[f"{name}_total"] = sum(times)
                stats[f"{name}_count"] = len(times)
        
        if self.memory_usage:
            stats["memory_avg_mb"] = sum(self.memory_usage) / len(self.memory_usage)
            stats["memory_peak_mb"] = max(self.memory_usage)
        
        return stats

# 全局性能监控器
_global_monitor = PerformanceMonitor()

class Resizing(nn.Module):
    """
    A utility module to resize tensors to a target channel count.
    This is used to match channel dimensions when operations with different
    channel counts are mixed.
    """
    def __init__(self, C_in, C_out, affine=True):
        super(Resizing, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )
    
    def forward(self, x):
        if self.C_in == self.C_out:
            return x
        return self.op(x)

class GatedCell(nn.Module):
    """A cell wrapped in a gate for dynamic depth."""
    def __init__(self, cell, C_in, C_out):
        super(GatedCell, self).__init__()
        self.cell = cell
        self.gate = nn.Parameter(1e-3 * torch.randn(1))
        self.resize = Resizing(C_in, C_out) if C_in != C_out else None

    def forward(self, s0, s1, weights):
        cell_out = self.cell(s0, s1, weights)
        gated_out = self.gate.sigmoid() * cell_out
        
        # Adjust s1 to match output dimension if necessary
        s1_resized = self.resize(s1) if self.resize else s1
        
        return s1_resized + gated_out

class Cell(nn.Module):
    """
    A single cell in the network, represented as a DAG.
    This is the searchable version of the cell, using MixedOp.
    """

    def __init__(self, steps, block_multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction
        
        # 性能监控属性
        self._forward_count = 0
        self._step_times = []

        # In a reduction cell, the previous cell's output is down-sampled
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._block_multiplier = block_multiplier

        self._ops = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)
                self._ops.append(op)

    def forward(self, s0, s1, weights):
        """增强的前向传播，包含详细进度跟踪"""
        self._forward_count += 1
        start_time = time.perf_counter()
        
        # 预处理阶段
        _global_monitor.start_timer("cell_preprocess")
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        preprocess_time = _global_monitor.end_timer("cell_preprocess")

        states = [s0, s1]
        offset = 0
        
        # 逐步计算每个中间节点
        for i in range(self._steps):
            _global_monitor.start_timer(f"cell_step_{i}")
            
            # 并行计算所有连接
            node_outputs = []
            for j, h in enumerate(states):
                op_idx = offset + j
                if op_idx < len(self._ops):
                    weight_idx = offset + j
                    if weight_idx < len(weights):
                        op_output = self._ops[op_idx](h, weights[weight_idx])
                        node_outputs.append(op_output)
            
            # 求和所有输入
            if node_outputs:
                s = sum(node_outputs)
            else:
                s = torch.zeros_like(states[0])
            
            step_time = _global_monitor.end_timer(f"cell_step_{i}")
            self._step_times.append(step_time)
            
            offset += len(states)
            states.append(s)

        # 最终输出
        _global_monitor.start_timer("cell_concat")
        result = torch.cat(states[-self._block_multiplier:], dim=1)
        concat_time = _global_monitor.end_timer("cell_concat")
        
        total_time = time.perf_counter() - start_time
        
        # 清理内存（关闭性能输出）
        if self._forward_count % 200 == 0:
            torch.cuda.empty_cache()
            if self._forward_count % 500 == 0:
                gc.collect()

        return result

class Network(nn.Module):
    """
    The full neural network model, composed of a stack of cells.
    This class also initializes and stores the architecture parameters (alphas).
    This version is simplified to remove all optimization flags.
    """

    def __init__(self, C, num_classes, layers, steps=4, block_multiplier=4, stem_multiplier=3, quiet=False):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._steps = steps
        self._block_multiplier = block_multiplier
        self.quiet = quiet

        if not quiet:
            print("🏗️  Building Search Network...")

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            
            cell = Cell(steps, block_multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, self._block_multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()

    def new(self):
        """Create a new model with the same architecture but uninitialized weights."""
        model_new = Network(self._C, self._num_classes, self._layers, self._steps, self._block_multiplier, quiet=True).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, input):
        weights_normal = F.softmax(self.alphas_normal, dim=-1)
        weights_reduce = F.softmax(self.alphas_reduce, dim=-1)

        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = weights_reduce
            else:
                weights = weights_normal
            s0, s1 = s1, cell(s0, s1, weights)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def _initialize_alphas(self):
        """Initialize the architecture parameters alpha."""
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        self.alphas_normal = nn.Parameter(1e-3 * torch.randn(k, num_ops))
        self.alphas_reduce = nn.Parameter(1e-3 * torch.randn(k, num_ops))
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):
        """
        Decodes the learned architecture parameters (alphas) into a discrete genotype.
        This method determines the final architecture of the network after search.
        """

        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                
                # For each node, find the two strongest predecessor connections
                edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')) if any(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')) else -1.0)[:2]
                
                # For each of the two selected edges, find the best operation
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    # Default to a safe choice if all options are zero
                    if k_best is None:
                        k_best = 1 # 'max_pool_3x3' as a fallback
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        # Normalize the alpha values using softmax
        gene_normal = _parse(torch.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
        gene_reduce = _parse(torch.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

        concat = range(2 + self._steps - self._block_multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype 