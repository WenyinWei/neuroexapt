
import torch
import torch.nn as nn
import time
import psutil
import gc
from typing import Dict, List, Optional
from .operations import OPS, FactorizedReduce, ReLUConvBN, MixedOp, Resizing, OptimizedMixedOp, LazyMixedOp, GradientOptimizedMixedOp, MemoryEfficientMixedOp
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

class Cell(nn.Module):
    """
    A single cell in the network, represented as a DAG.
    Each cell consists of a set of nodes, where each node computes a feature map
    by applying mixed operations to the feature maps of its predecessors.
    """

    def __init__(self, steps, block_multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, 
                 use_optimized_ops=False, use_lazy_ops=False, use_gradient_optimized=False, use_memory_efficient=False):
        super(Cell, self).__init__()
        self.reduction = reduction
        self.use_optimized_ops = use_optimized_ops
        self.use_lazy_ops = use_lazy_ops
        self.use_gradient_optimized = use_gradient_optimized
        self.use_memory_efficient = use_memory_efficient

        # In a reduction cell, the previous cell's output is down-sampled
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._block_multiplier = block_multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                
                # 选择MixedOp类型，优先级：gradient_optimized > memory_efficient > lazy > optimized > standard
                if use_gradient_optimized:
                    op = GradientOptimizedMixedOp(C, stride)
                elif use_memory_efficient:
                    op = MemoryEfficientMixedOp(C, stride)
                elif use_lazy_ops:
                    op = LazyMixedOp(C, stride)
                elif use_optimized_ops:
                    op = OptimizedMixedOp(C, stride)
                else:
                    op = MixedOp(C, stride)
                self._ops.append(op)
        
        # 进度跟踪
        self._forward_count = 0
        self._step_times: List[float] = []

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
    """

    def __init__(self, C, num_classes, layers, potential_layers=4, steps=4, block_multiplier=4, *, 
                 use_checkpoint: bool = False, use_compile: bool = False, compile_backend: str = "inductor",
                 use_optimized_ops: bool = False, use_lazy_ops: bool = False, 
                 use_gradient_optimized: bool = False, use_memory_efficient: bool = False,
                 progress_tracking: bool = True, quiet: bool = False):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._potential_layers = potential_layers
        self._steps = steps
        self._block_multiplier = block_multiplier
        self.use_checkpoint = use_checkpoint
        self.use_compile = use_compile
        self.use_optimized_ops = use_optimized_ops
        self.use_lazy_ops = use_lazy_ops
        self.use_gradient_optimized = use_gradient_optimized
        self.use_memory_efficient = use_memory_efficient
        self.progress_tracking = progress_tracking
        self.quiet = quiet

        # 网络结构构建进度（简化输出）
        if not quiet and progress_tracking:
            print(f"🏗️  构建网络架构...")
            print(f"   基础层数: {layers}, 潜在层数: {potential_layers}")
            optimizations = []
            if use_gradient_optimized:
                optimizations.append("梯度优化")
            if use_lazy_ops:
                optimizations.append("懒计算")
            if use_memory_efficient:
                optimizations.append("内存优化")
            if optimizations:
                print(f"   启用优化: {', '.join(optimizations)}")

        C_curr = self._block_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        total_layers = layers + potential_layers
        
        # 只显示总数，不逐个显示cell
        if not quiet and progress_tracking:
            print(f"   📐 创建 {total_layers} 个Cell...")
        
        for i in range(total_layers):
            # Reduction cells are placed based on the initial layer count
            if i < layers and i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            
            # For potential layers, ensure they don't break the channel progression
            if i >= layers:
                # Keep potential layers at the same channel count as the last regular layer
                # to avoid shape mismatches
                reduction = False
            
            cell = Cell(steps, block_multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, 
                       use_optimized_ops, use_lazy_ops, use_gradient_optimized, use_memory_efficient)
            
            # Wrap potential layers in a GatedCell
            if i >= layers:
                cell = GatedCell(cell, C_prev, C_curr * self._block_multiplier)

            reduction_prev = reduction
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, self._block_multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()

        # 网络前向传播计数
        self._forward_count = 0
        self._epoch_forward_count = 0

        # Optionally compile the full model (PyTorch 2.0+)
        if self.use_compile and hasattr(torch, "compile"):
            if not quiet:
                print(f"   ⚡ 启用torch.compile优化...")
            # torch.compile returns a new compiled module; swap forward to point to it
            compiled_self = torch.compile(self, backend=compile_backend, fullgraph=False)
            # Keep reference to avoid GC
            self._compiled_impl = compiled_self
            self.forward = compiled_self.forward  # type: ignore[method-assign]

        if not quiet and progress_tracking:
            print(f"✅ 网络构建完成!")

    def new(self):
        """Create a new model with the same architecture but uninitialized weights."""
        model_new = Network(self._C, self._num_classes, self._layers, self._potential_layers, 
                          use_checkpoint=self.use_checkpoint, use_compile=self.use_compile, 
                          compile_backend="inductor", use_optimized_ops=self.use_optimized_ops,
                          quiet=True).cuda()  # 新建模型时默认安静模式
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, input):
        """增强的前向传播，包含详细进度跟踪"""
        self._forward_count += 1
        self._epoch_forward_count += 1
        
        _global_monitor.start_timer("network_forward")

        # Precompute softmax-ed architecture weights once per forward pass
        _global_monitor.start_timer("softmax_weights")
        weights_normal = torch.softmax(self.alphas_normal, dim=-1)
        weights_reduce = torch.softmax(self.alphas_reduce, dim=-1)
        weights_time = _global_monitor.end_timer("softmax_weights")

        import torch.utils.checkpoint as cp

        # Stem处理
        _global_monitor.start_timer("stem")
        s0 = s1 = self.stem(input)
        stem_time = _global_monitor.end_timer("stem")
        
        # 关闭Stem输出
        # if self.progress_tracking and self._forward_count % 200 == 1:
        #     print(f"     ✅ Stem完成: {stem_time*1000:.2f}ms")

        # 逐层处理（仅在非安静模式且启用进度跟踪时显示）
        if not self.quiet and self.progress_tracking and self._forward_count <= 3:  # 只在前几次forward时显示
            print(f"  🔗 开始处理 {len(self.cells)} 个Cell...")
        
        for i, cell in enumerate(self.cells):
            layer_start = time.perf_counter()
            
            # 只在非安静模式、启用进度跟踪且前几次forward时显示详细信息
            if not self.quiet and self.progress_tracking and self._forward_count <= 2:
                cell_type = "GatedCell" if isinstance(cell, GatedCell) else "Cell"
                is_reduction = (isinstance(cell, GatedCell) and cell.cell.reduction) or (hasattr(cell, 'reduction') and cell.reduction)
                reduction_info = "Reduction" if is_reduction else "Normal"
                print(f"    🏭 第{i+1}/{len(self.cells)}层 [{cell_type}-{reduction_info}]...")

            # Determine which set of precomputed weights to use
            if isinstance(cell, GatedCell):
                if cell.cell.reduction:
                    weights = weights_reduce
                else:
                    weights = weights_normal
            else:
                weights = weights_reduce if cell.reduction else weights_normal

            _global_monitor.start_timer(f"layer_{i}")

            if self.use_checkpoint and self.training:
                # Wrap cell forward in checkpoint to save memory
                def _cell_forward(a, b):
                    return cell(a, b, weights)

                s1_new = cp.checkpoint(_cell_forward, s0, s1)
                checkpoint_info = " (checkpointed)"
            else:
                s1_new = cell(s0, s1, weights)
                checkpoint_info = ""

            layer_time = _global_monitor.end_timer(f"layer_{i}")

            # 关闭层级输出
            # if self.progress_tracking and (i % 6 == 0 or self._forward_count % 50 == 1):
            #     shape_info = s1_new.shape if hasattr(s1_new, 'shape') else 'unknown'
            #     print(f"       ✅ 第{i+1}层完成: {layer_time*1000:.2f}ms{checkpoint_info}")

            s0, s1 = s1, s1_new
        
        # 全局池化和分类
        _global_monitor.start_timer("classification")
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        classification_time = _global_monitor.end_timer("classification")

        total_time = _global_monitor.end_timer("network_forward")
        
        # 关闭前向传播输出
        # if self.progress_tracking and self._forward_count % 50 == 1:
        #     print(f"  📊 前向传播完成: {total_time*1000:.2f}ms")
        
        # 关闭详细统计输出
        # if self._forward_count % 200 == 0:
        #     stats = _global_monitor.get_stats()
        #     torch.cuda.empty_cache()
        #     gc.collect()

        return logits

    def reset_epoch_counters(self):
        """重置epoch计数器"""
        self._epoch_forward_count = 0
        if not self.quiet and self.progress_tracking:
            print(f"🔄 重置epoch计数器")

    def _initialize_alphas(self):
        """Initialize the architecture parameters alpha."""
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(OPS)

        self.alphas_normal = nn.Parameter(1e-3 * torch.randn(k, num_ops))
        self.alphas_reduce = nn.Parameter(1e-3 * torch.randn(k, num_ops))
        
        # Initialize gates for potential cells
        self.alphas_gates = nn.ParameterList(
            [cell.gate for cell in self.cells if isinstance(cell, GatedCell)]
        )

        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]
        # Register the gate parameters with the architect
        self._arch_parameters.extend(self.alphas_gates)

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

class GatedCell(nn.Module):
    """
    A wrapper for a cell that includes a learnable gate to control its contribution.
    This version implements a function-preserving residual connection, ensuring that
    when the gate is closed, it acts as an identity connection.
    """
    def __init__(self, cell, C_in_s1, C_out_cell):
        super(GatedCell, self).__init__()
        self.cell = cell
        self.gate = nn.Parameter(torch.randn(1) * 1e-3)
        # Resizer for the identity path to match the cell's output dimensions
        self.identity_resizer = Resizing(C_in_s1, C_out_cell, affine=False)

    def forward(self, s0, s1, weights):
        """
        Computes: gate * cell(s0, s1) + (1 - gate) * identity(s1)
        """
        gate_val = torch.sigmoid(self.gate)
        
        cell_out = self.cell(s0, s1, weights)
        identity_out = self.identity_resizer(s1)
        
        # 确保形状匹配
        if cell_out.shape != identity_out.shape:
            # 使用identity输出的形状作为目标
            cell_out = self.identity_resizer(s1)
        
        return gate_val * cell_out + (1 - gate_val) * identity_out 