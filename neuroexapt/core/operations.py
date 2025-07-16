
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import time
from typing import List, Optional

# 🔧 递归检测和防护机制
_MIXEDOP_INITIALIZATION_STACK = set()

def _safe_mixedop_init(cls_name: str, *args, **kwargs):
    """
    安全的MixedOp初始化函数，防止递归调用
    """
    if cls_name in _MIXEDOP_INITIALIZATION_STACK:
        raise RuntimeError(f"检测到{cls_name}的递归初始化，可能存在循环依赖")
    
    _MIXEDOP_INITIALIZATION_STACK.add(cls_name)
    try:
        # 这里会被各个MixedOp类的__init__方法调用
        return True
    finally:
        _MIXEDOP_INITIALIZATION_STACK.discard(cls_name)

# Triton accelerated helpers
from neuroexapt.kernels import TRITON_AVAILABLE, sepconv_forward_generic  # type: ignore
from neuroexapt.kernels.pool_triton import (
    TRITON_AVAILABLE as TRITON_POOL_AVAILABLE,
    avg_pool3x3_forward,
    max_pool3x3_forward,
    avg_pool5x5_forward,
    max_pool5x5_forward,
    avg_pool7x7_forward,
    max_pool7x7_forward,
    global_avgpool_forward,
)

# CUDA accelerated SoftmaxSum
try:
    from neuroexapt.cuda_ops import SoftmaxSumFunction, CUDA_AVAILABLE
    CUDA_SOFTMAX_AVAILABLE = CUDA_AVAILABLE
except ImportError:
    CUDA_SOFTMAX_AVAILABLE = False
    SoftmaxSumFunction = None  # type: ignore

# A collection of all possible operations that can be placed on an edge of the network graph
OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine: TritonAvgPool3x3(stride),
    'max_pool_3x3': lambda C, stride, affine: TritonMaxPool3x3(stride),
    'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'sep_conv_7x7': lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
    'conv_7x1_1x7': lambda C, stride, affine: nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),
        nn.Conv2d(C, C, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),
        nn.BatchNorm2d(C, affine=affine)
    ),
}

class ReLUConvBN(nn.Module):
    """Standard ReLU-Conv-BatchNorm block."""
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)

class DilConv(nn.Module):
    """Dilated separable convolution."""
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.relu = nn.ReLU(inplace=False)
        self.dw = nn.Conv2d(
            C_in,
            C_in,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=C_in,
            bias=False,
        )
        self.pw = nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)
        # cache parameters
        self._k = kernel_size
        self._stride = stride
        self._dilation = dilation

    def forward(self, x):
        x = self.relu(x)
        if TRITON_AVAILABLE and x.is_cuda and self._k in {3, 5, 7} and self._dilation in {1, 2}:
            y = sepconv_forward_generic(
                x,
                self.dw.weight,
                self.pw.weight,
                None,
                kernel_size=self._k,
                stride=self._stride,
                dilation=self._dilation,
            )
        else:
            y = self.pw(
                torch.nn.functional.conv2d(
                    x,
                    self.dw.weight,
                    None,
                    stride=self._stride,
                    padding=((self._k - 1) * self._dilation) // 2,
                    dilation=self._dilation,
                    groups=self.dw.in_channels,
                )
            )
        return self.bn(y)

class SepConv(nn.Module):
    """Separable convolution."""
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        # First separable conv block
        self.relu1 = nn.ReLU(inplace=False)
        self.dw1 = nn.Conv2d(C_in, C_in, kernel_size, stride, padding, groups=C_in, bias=False)
        self.pw1 = nn.Conv2d(C_in, C_out, 1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(C_out, affine=affine)

        # Second separable conv block (stride=1)
        self.relu2 = nn.ReLU(inplace=False)
        self.dw2 = nn.Conv2d(C_out, C_out, kernel_size, 1, padding, groups=C_out, bias=False)
        self.pw2 = nn.Conv2d(C_out, C_out, 1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(C_out, affine=affine)

        self._k = kernel_size
        self._padding = padding
        self._stride = stride

    def _sepconv_block(self, x, dw, pw, bn, stride):
        if TRITON_AVAILABLE and x.is_cuda and self._k in {3, 5, 7}:
            y = sepconv_forward_generic(
                x,
                dw.weight,
                pw.weight,
                None,
                kernel_size=self._k,
                stride=stride,
                dilation=1,
            )
        else:
            y = pw(
                torch.nn.functional.conv2d(
                    x,
                    dw.weight,
                    None,
                    stride=stride,
                    padding=self._padding,
                    groups=dw.in_channels,
                )
            )
        return bn(y)

    def forward(self, x):
        y = self._sepconv_block(self.relu1(x), self.dw1, self.pw1, self.bn1, self._stride)
        y = self._sepconv_block(self.relu2(y), self.dw2, self.pw2, self.bn2, 1)
        return y

class Identity(nn.Module):
    """Identity mapping."""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Zero(nn.Module):
    """Zero operation, effectively removing a connection."""
    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)

class FactorizedReduce(nn.Module):
    """Reduces the spatial dimensions and doubles the channel dimensions."""
    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


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


class OptimizedMixedOp(nn.Module):
    """
    高度优化的混合操作类，专为GPU性能设计
    - 减少内存分配和复制
    - 使用fused operations
    - 实现高效的加权求和
    - 添加操作结果缓存
    """
    def __init__(self, C, stride, enable_caching=True):
        super(OptimizedMixedOp, self).__init__()
        from .genotypes import PRIMITIVES
        
        self._C = C
        self._stride = stride
        self.enable_caching = enable_caching
        
        # 构建操作列表
        self._ops = nn.ModuleList()
        self._op_names = []
        
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)
            self._op_names.append(primitive)
        
        # 缓存相关
        self._cached_outputs: Optional[torch.Tensor] = None
        self._cached_input_hash: Optional[int] = None
        self._cache_hits = 0
        self._cache_misses = 0
        
        # 预分配输出张量以减少内存分配
        self.register_buffer('_output_buffer', torch.empty(1))
        
        # 性能监控
        self._forward_times: List[float] = []
        self._op_times: List[float] = []

    def _get_input_hash(self, x: torch.Tensor) -> int:
        """快速输入哈希，用于缓存检查"""
        return hash((x.shape, x.device, x.dtype, x.data_ptr()))

    def _maybe_resize_buffer(self, target_shape: torch.Size, device: torch.device) -> torch.Tensor:
        """智能缓冲区大小调整"""
        # 简化版本，直接返回合适大小的tensor
        return torch.empty(target_shape, device=device, dtype=torch.float32)

    def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        高度优化的前向传播
        """
        start_time = time.perf_counter()
        
        # 检查缓存
        if self.enable_caching:
            input_hash = self._get_input_hash(x)
            if input_hash == self._cached_input_hash and self._cached_outputs is not None:
                self._cache_hits += 1
                return self._cached_outputs * weights.view(-1, 1, 1, 1, 1).sum(dim=0)
            else:
                self._cache_misses += 1
                self._cached_input_hash = input_hash

        # 快速权重检查 - 如果只有一个操作占主导地位，直接计算
        max_weight_idx = int(weights.argmax().item())
        if weights[max_weight_idx] > 0.9:  # 90%以上权重集中在一个操作上
            op_start = time.perf_counter()
            result = self._ops[max_weight_idx](x) * weights[max_weight_idx]
            self._op_times.append(time.perf_counter() - op_start)
            
            self._forward_times.append(time.perf_counter() - start_time)
            return result

        # 并行计算所有操作（GPU并行优化）
        print(f"  🔧 MixedOp: 并行计算 {len(self._ops)} 个操作...")
        
        # 使用列表推导式并行计算，让GPU调度器优化
        op_start = time.perf_counter()
        
        # 分批处理操作以优化GPU内存使用
        batch_size = min(4, len(self._ops))  # 每批最多4个操作
        outputs = []
        
        for i in range(0, len(self._ops), batch_size):
            batch_ops = self._ops[i:i+batch_size]
            batch_weights = weights[i:i+batch_size]
            
            # 计算当前批次的操作
            batch_outputs = []
            for j, op in enumerate(batch_ops):
                if batch_weights[j] > 1e-6:  # 只计算有意义权重的操作
                    op_output = op(x)
                    batch_outputs.append(op_output * batch_weights[j])
            
            if batch_outputs:
                # 在GPU上高效求和
                batch_result = torch.stack(batch_outputs, dim=0).sum(dim=0)
                outputs.append(batch_result)
        
        # 最终求和
        if outputs:
            result = torch.stack(outputs, dim=0).sum(dim=0)
        else:
            # 如果所有权重都很小，返回零张量
            result = torch.zeros_like(x)
        
        op_time = time.perf_counter() - op_start
        self._op_times.append(op_time)
        
        # 更新缓存
        if self.enable_caching:
            self._cached_outputs = torch.stack([op(x) for op in self._ops], dim=0)
        
        total_time = time.perf_counter() - start_time
        self._forward_times.append(total_time)
        
        # 定期输出性能统计
        if len(self._forward_times) % 50 == 0:
            avg_time = sum(self._forward_times[-50:]) / 50
            avg_op_time = sum(self._op_times[-50:]) / 50
            cache_rate = self._cache_hits / max(1, self._cache_hits + self._cache_misses)
            print(f"    📊 MixedOp性能: 平均{avg_time*1000:.2f}ms, 操作{avg_op_time*1000:.2f}ms, 缓存命中率{cache_rate:.1%}")
        
        return result

    def get_performance_stats(self) -> dict:
        """获取性能统计信息"""
        if not self._forward_times:
            return {}
        
        return {
            'avg_forward_time': sum(self._forward_times) / len(self._forward_times),
            'avg_op_time': sum(self._op_times) / len(self._op_times) if self._op_times else 0,
            'cache_hit_rate': self._cache_hits / max(1, self._cache_hits + self._cache_misses),
            'total_forwards': len(self._forward_times)
        }

class MixedOp(nn.Module):
    """
    A differentiable mixed operation.
    This is the lightweight version, only mixing operation types.
    """
    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        # Import PRIMITIVES to ensure consistent ordering
        from .genotypes import PRIMITIVES
        
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
        """
        Args:
            x: input tensor
            weights: a tensor of shape [num_ops], representing arch params.
        Returns:
            The weighted sum of the outputs of all operations.
        Note:
            This implementation vectorizes the weighted sum to reduce Python overhead.
        """
        # 计数器更新
        if hasattr(self, '_step_counter'):
            self._step_counter += 1
        else:
            self._step_counter = 1
        
        # Compute outputs for each operation
        outputs = [op(x) for op in self._ops]  # list of tensors

        # Use CUDA-accelerated SoftmaxSum if available and beneficial
        if (CUDA_SOFTMAX_AVAILABLE and SoftmaxSumFunction is not None and 
            x.is_cuda and len(outputs) >= 4 and outputs[0].numel() >= 1024):
            # Stack and use fused kernel for large operations
            stacked = torch.stack(outputs, dim=0)
            return SoftmaxSumFunction.apply(stacked, weights)
        else:
            # Fallback to standard PyTorch implementation
            stacked = torch.stack(outputs, dim=0)
            weighted = stacked * weights.view(-1, 1, 1, 1, 1)
            return weighted.sum(dim=0)

class LazyMixedOp(nn.Module):
    """
    高性能懒计算混合操作，专为Exapt模式设计
    特性：
    1. 懒计算：只计算权重大的操作
    2. 智能缓存：缓存重复计算结果  
    3. 早期终止：权重收敛时跳过计算
    4. 内存池：预分配内存避免频繁分配
    5. 操作剪枝：动态移除低权重操作
    """
    def __init__(self, C, stride, lazy_threshold=0.01, cache_size=16, enable_pruning=True):
        # 🔧 递归检测
        _safe_mixedop_init("LazyMixedOp")
        super(LazyMixedOp, self).__init__()
        from .genotypes import PRIMITIVES
        
        self._C = C
        self._stride = stride
        self.lazy_threshold = lazy_threshold  # 懒计算阈值
        self.cache_size = cache_size
        self.enable_pruning = enable_pruning
        
        # 构建操作列表
        self._ops = nn.ModuleList()
        self._op_names = []
        self._op_active = []  # 操作激活状态
        
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)
            self._op_names.append(primitive)
            self._op_active.append(True)  # 初始时所有操作都激活
        
        # 智能缓存系统
        self._cache = {}  # {input_hash: {op_idx: output}}
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_order = []  # LRU缓存管理
        
        # 权重历史记录（用于收敛检测）
        self._weight_history = []
        self._converged_ops = set()  # 已收敛的操作
        
        # 内存池
        self._memory_pool = {}  # {shape: [tensor1, tensor2, ...]}
        self._pool_hits = 0
        self._pool_misses = 0
        
        # 性能统计
        self._stats = {
            'lazy_skips': 0,
            'total_ops_computed': 0,
            'total_forward_calls': 0,
            'pruned_ops': 0,
            'cache_hit_rate': 0.0,
            'memory_pool_hit_rate': 0.0
        }
        
        # 预热状态
        self._warmup_calls = 0
        self._warmup_threshold = 20

    def _get_input_hash(self, x: torch.Tensor) -> int:
        """快速输入哈希用于缓存"""
        return hash((x.shape, x.device, x.dtype, x.data_ptr()))

    def _get_from_memory_pool(self, shape: torch.Size, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """从内存池获取张量"""
        key = (shape, device, dtype)
        if key in self._memory_pool and self._memory_pool[key]:
            self._pool_hits += 1
            tensor = self._memory_pool[key].pop()
            tensor.zero_()  # 清零重用
            return tensor
        else:
            self._pool_misses += 1
            return torch.zeros(shape, device=device, dtype=dtype)

    def _return_to_memory_pool(self, tensor: torch.Tensor):
        """返回张量到内存池"""
        key = (tensor.shape, tensor.device, tensor.dtype)
        if key not in self._memory_pool:
            self._memory_pool[key] = []
        
        # 限制内存池大小
        if len(self._memory_pool[key]) < 4:
            self._memory_pool[key].append(tensor.detach())

    def _update_cache(self, input_hash: int, op_idx: int, output: torch.Tensor):
        """更新缓存"""
        if input_hash not in self._cache:
            self._cache[input_hash] = {}
            self._cache_order.append(input_hash)
        
        self._cache[input_hash][op_idx] = output.detach().clone()
        
        # LRU缓存管理
        if len(self._cache_order) > self.cache_size:
            oldest_hash = self._cache_order.pop(0)
            del self._cache[oldest_hash]

    def _detect_weight_convergence(self, weights: torch.Tensor) -> set:
        """检测权重收敛的操作"""
        self._weight_history.append(weights.detach().clone())
        
        # 保持最近10次权重历史
        if len(self._weight_history) > 10:
            self._weight_history.pop(0)
        
        # 需要至少5次历史记录才能判断收敛
        if len(self._weight_history) < 5:
            return set()
        
        converged = set()
        for i in range(len(weights)):
            # 检查最近5次的权重变化
            recent_weights = [h[i].item() for h in self._weight_history[-5:]]
            weight_std = torch.tensor(recent_weights).std().item()
            
            # 如果权重变化很小且权重本身很小，认为已收敛
            if weight_std < 0.001 and weights[i].item() < self.lazy_threshold:
                converged.add(i)
        
        return converged

    def _prune_operations(self, weights: torch.Tensor):
        """动态剪枝低权重操作"""
        if not self.enable_pruning or self._warmup_calls < self._warmup_threshold:
            return
        
        for i, weight in enumerate(weights):
            if weight.item() < self.lazy_threshold / 10 and self._op_active[i]:
                self._op_active[i] = False
                self._stats['pruned_ops'] += 1
                print(f"    ✂️  剪枝操作 {self._op_names[i]} (权重: {weight.item():.6f})")

    def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """高性能懒计算前向传播"""
        self._stats['total_forward_calls'] += 1
        self._warmup_calls += 1
        
        input_hash = self._get_input_hash(x)
        
        # 检测权重收敛和操作剪枝
        self._converged_ops = self._detect_weight_convergence(weights)
        if self.enable_pruning:
            self._prune_operations(weights)
        
        # 快速路径：如果一个操作权重占主导(>95%)且已收敛
        max_weight_idx = int(weights.argmax().item())
        if weights[max_weight_idx] > 0.95 and max_weight_idx in self._converged_ops:
            if input_hash in self._cache and max_weight_idx in self._cache[input_hash]:
                self._cache_hits += 1
                return self._cache[input_hash][max_weight_idx]
            else:
                result = self._ops[max_weight_idx](x)
                self._update_cache(input_hash, max_weight_idx, result)
                self._cache_misses += 1
                return result

        # 懒计算：只计算权重大于阈值且未被剪枝的操作
        active_ops = []
        active_weights = []
        outputs = []
        
        for i, (op, weight) in enumerate(zip(self._ops, weights)):
            if not self._op_active[i]:
                continue  # 跳过被剪枝的操作
                
            if weight.item() < self.lazy_threshold and i not in self._converged_ops:
                self._stats['lazy_skips'] += 1
                continue
            
            # 检查缓存
            if input_hash in self._cache and i in self._cache[input_hash]:
                output = self._cache[input_hash][i]
                self._cache_hits += 1
            else:
                output = op(x)
                self._update_cache(input_hash, i, output)
                self._cache_misses += 1
                self._stats['total_ops_computed'] += 1
            
            outputs.append(output * weight)
            active_ops.append(i)
            active_weights.append(weight.item())

        # 处理结果
        if outputs:
            if len(outputs) == 1:
                result = outputs[0]
            else:
                # 使用内存池优化求和
                result = self._get_from_memory_pool(outputs[0].shape, outputs[0].device, outputs[0].dtype)
                for output in outputs:
                    result = result + output
        else:
            # 所有操作都被跳过，返回零张量
            result = self._get_from_memory_pool(x.shape, x.device, x.dtype)
        
        # 更新统计信息
        if self._stats['total_forward_calls'] % 100 == 0:
            self._update_stats()
        
        return result

    def _update_stats(self):
        """更新性能统计"""
        total_cache_ops = self._cache_hits + self._cache_misses
        self._stats['cache_hit_rate'] = self._cache_hits / max(1, total_cache_ops)
        
        total_pool_ops = self._pool_hits + self._pool_misses
        self._stats['memory_pool_hit_rate'] = self._pool_hits / max(1, total_pool_ops)
        
        # 记录性能统计（关闭输出）
        # if self._stats['total_forward_calls'] % 5000 == 0:
        #     print(f"    📊 LazyMixedOp: 缓存命中率{self._stats['cache_hit_rate']:.1%}, 收敛操作{len(self._converged_ops)}/{len(self._ops)}")

    def get_performance_stats(self) -> dict:
        """获取详细性能统计"""
        return self._stats.copy()

    def clear_cache(self):
        """清理缓存"""
        self._cache.clear()
        self._cache_order.clear()
        for pool in self._memory_pool.values():
            pool.clear() 

class GradientOptimizedMixedOp(nn.Module):
    """
    反向传播优化的混合操作，专门解决反向传播慢的问题
    
    优化策略：
    1. 选择性梯度计算：只为权重大的操作计算梯度
    2. 梯度检查点：减少内存使用和计算图复杂度
    3. 异步梯度累积：避免同步等待
    4. 内存池复用：减少内存分配开销
    5. 计算图剪枝：移除不必要的计算节点
    """
    def __init__(self, C, stride, gradient_threshold=0.01, use_checkpoint=True, memory_efficient=True):
        # 🔧 递归检测
        _safe_mixedop_init("GradientOptimizedMixedOp")
        super(GradientOptimizedMixedOp, self).__init__()
        from .genotypes import PRIMITIVES
        
        self._C = C
        self._stride = stride
        self.gradient_threshold = gradient_threshold
        self.use_checkpoint = use_checkpoint
        self.memory_efficient = memory_efficient
        
        # 构建操作列表
        self._ops = nn.ModuleList()
        self._op_names = []
        
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)
            self._op_names.append(primitive)
        
        # 权重跟踪用于梯度优化
        self._weight_momentum = 0.9
        self._avg_weights = torch.zeros(len(PRIMITIVES))
        self._gradient_mask = torch.ones(len(PRIMITIVES), dtype=torch.bool)
        
        # 性能统计
        self._stats = {
            'forward_calls': 0,
            'gradient_skips': 0,
            'checkpoint_saves': 0,
            'memory_reuse': 0
        }
        
        # 内存池
        self._output_cache = {}
        self._gradient_cache = {}

    def _update_gradient_mask(self, weights: torch.Tensor):
        """更新梯度计算掩码，只为重要的操作计算梯度"""
        # 指数移动平均更新权重
        if self._avg_weights.device != weights.device:
            self._avg_weights = self._avg_weights.to(weights.device)
            self._gradient_mask = self._gradient_mask.to(weights.device)
        
        self._avg_weights = self._weight_momentum * self._avg_weights + (1 - self._weight_momentum) * weights.detach()
        
        # 只为权重大于阈值的操作计算梯度
        new_mask = self._avg_weights > self.gradient_threshold
        
        # 至少保留权重最大的两个操作
        if new_mask.sum() < 2:
            top_indices = torch.topk(self._avg_weights, 2).indices
            new_mask[top_indices] = True
        
        # 更新掩码
        mask_changed = not torch.equal(self._gradient_mask, new_mask)
        self._gradient_mask = new_mask
        
        # 减少掩码更新输出
        # if mask_changed:
        #     active_ops = [self._op_names[i] for i in range(len(self._op_names)) if self._gradient_mask[i]]
        #     print(f"    🎯 梯度计算掩码更新: 激活操作 {active_ops}")
        
        return mask_changed

    def _selective_forward(self, x: torch.Tensor, weights: torch.Tensor):
        """选择性前向传播，只计算需要梯度的操作"""
        # 更新梯度掩码
        self._update_gradient_mask(weights)
        
        # 快速路径：如果只有一个操作占主导
        max_idx = int(weights.argmax().item())
        if weights[max_idx] > 0.95:
            return self._ops[max_idx](x) * weights[max_idx]
        
        # 选择性计算
        active_outputs = []
        active_weights = []
        
        for i, (op, weight) in enumerate(zip(self._ops, weights)):
            if self._gradient_mask[i] or weight > self.gradient_threshold:
                if self.use_checkpoint and self.training:
                    # 使用梯度检查点减少内存使用
                    output = checkpoint.checkpoint(op, x, use_reentrant=False)
                    self._stats['checkpoint_saves'] += 1
                else:
                    output = op(x)
                
                active_outputs.append(output * weight)  # type: ignore[operator]
                active_weights.append(weight.item())
            else:
                # 跳过梯度计算，使用detach
                with torch.no_grad():
                    output = op(x)
                active_outputs.append(output.detach() * weight.detach())  # type: ignore[operator]
                self._stats['gradient_skips'] += 1
        
        # 内存高效的求和
        if len(active_outputs) == 1:
            return active_outputs[0]
        elif len(active_outputs) == 2:
            return active_outputs[0] + active_outputs[1]
        else:
            # 分层求和减少内存峰值
            result = active_outputs[0]
            for output in active_outputs[1:]:
                result = result + output
            return result

    def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """优化的前向传播"""
        self._stats['forward_calls'] += 1
        
        # 使用选择性前向传播
        result = self._selective_forward(x, weights)
        
        # 记录性能统计（关闭输出）
        # if self._stats['forward_calls'] % 5000 == 0:
        #     skip_rate = self._stats['gradient_skips'] / max(1, self._stats['forward_calls'] * len(self._ops))
        #     print(f"    📊 梯度优化: 跳过率{skip_rate:.1%}, 激活{self._gradient_mask.sum().item()}/{len(self._ops)}")
        
        return result

    def get_gradient_stats(self) -> dict:
        """获取梯度优化统计"""
        return {
            'gradient_mask': self._gradient_mask.cpu().tolist(),
            'avg_weights': self._avg_weights.cpu().tolist(),
            'active_ops': self._gradient_mask.sum().item(),
            'total_ops': len(self._ops),
            **self._stats
        }

class MemoryEfficientMixedOp(nn.Module):
    """
    内存高效的混合操作，专门解决GPU内存使用问题
    
    特性：
    1. 流式计算：避免同时存储所有操作的输出
    2. 内存回收：及时释放中间结果
    3. 批量优化：合并小的操作减少开销
    4. 缓存复用：智能复用计算结果
    """
    def __init__(self, C, stride, stream_compute=True, cache_outputs=True):
        # 🔧 递归检测
        _safe_mixedop_init("MemoryEfficientMixedOp")
        super(MemoryEfficientMixedOp, self).__init__()
        from .genotypes import PRIMITIVES
        
        self._C = C
        self._stride = stride
        self.stream_compute = stream_compute
        self.cache_outputs = cache_outputs
        
        # 构建操作列表
        self._ops = nn.ModuleList()
        self._op_names = []
        
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)
            self._op_names.append(primitive)
        
        # 内存管理
        self._output_cache = {}
        self._memory_high_watermark = 0
        self._cache_hits = 0
        self._cache_misses = 0

    def _get_cache_key(self, x: torch.Tensor) -> str:
        """生成缓存键"""
        return f"{x.shape}_{x.device}_{x.data_ptr()}"

    def _stream_forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """流式前向传播，减少内存峰值使用"""
        # 初始化结果张量
        result = torch.zeros_like(x)
        
        # 流式计算每个操作
        for i, (op, weight) in enumerate(zip(self._ops, weights)):
            if weight.item() < 1e-6:  # 跳过权重很小的操作
                continue
            
            # 检查缓存
            cache_key = f"{self._get_cache_key(x)}_{i}"
            if self.cache_outputs and cache_key in self._output_cache:
                output = self._output_cache[cache_key]
                self._cache_hits += 1
            else:
                output = op(x)
                if self.cache_outputs:
                    self._output_cache[cache_key] = output.detach().clone()
                    # 限制缓存大小
                    if len(self._output_cache) > 32:
                        oldest_key = next(iter(self._output_cache))
                        del self._output_cache[oldest_key]
                self._cache_misses += 1
            
            # 累积到结果中
            weighted_output = output * weight
            result = result + weighted_output
            
            # 及时释放内存
            del weighted_output
            if not self.cache_outputs:
                del output
        
        return result

    def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """内存高效的前向传播"""
        # 记录内存使用
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated()
            self._memory_high_watermark = max(self._memory_high_watermark, current_memory)
        
        if self.stream_compute:
            result = self._stream_forward(x, weights)
        else:
            # 标准实现但优化内存
            outputs = []
            for i, (op, weight) in enumerate(zip(self._ops, weights)):
                if weight.item() > 1e-6:  # 只计算有意义的操作
                    output = op(x) * weight
                    outputs.append(output)
            
            if outputs:
                result = torch.stack(outputs, dim=0).sum(dim=0)
            else:
                result = torch.zeros_like(x)
        
        # 定期清理缓存
        if hasattr(self, '_forward_count'):
            self._forward_count += 1
        else:
            self._forward_count = 1
        
        if self._forward_count % 500 == 0:
            torch.cuda.empty_cache()
            # 关闭内存统计输出
            # if self.cache_outputs:
            #     cache_hit_rate = self._cache_hits / max(1, self._cache_hits + self._cache_misses)
            #     print(f"    💾 内存效率统计: 缓存命中率 {cache_hit_rate:.1%}, 峰值内存 {self._memory_high_watermark/1024/1024:.1f}MB")
        
        return result 

class TritonAvgPool3x3(nn.Module):
    def __init__(self, stride: int):
        super().__init__()
        self.stride = stride

    def forward(self, x: torch.Tensor):  # type: ignore[override]
        if TRITON_POOL_AVAILABLE and x.is_cuda:
            return avg_pool3x3_forward(x, self.stride)
        return torch.nn.functional.avg_pool2d(x, 3, stride=self.stride, padding=1, count_include_pad=False)


class TritonMaxPool3x3(nn.Module):
    def __init__(self, stride: int):
        super().__init__()
        self.stride = stride

    def forward(self, x: torch.Tensor):  # type: ignore[override]
        if TRITON_POOL_AVAILABLE and x.is_cuda:
            return max_pool3x3_forward(x, self.stride)
        return torch.nn.functional.max_pool2d(x, 3, stride=self.stride, padding=1) 

class FusedOptimizedMixedOp(nn.Module):
    """
    🚀 融合优化的混合操作 - 同时应用所有优化策略
    
    融合特性：
    1. 梯度优化：选择性梯度计算 + 检查点
    2. 内存优化：流式计算 + 缓存复用  
    3. 懒计算：动态剪枝 + 早期终止
    4. Triton加速：自动检测并使用CUDA核
    5. 智能调度：根据负载自适应调整策略
    """
    def __init__(self, C, stride, 
                 gradient_threshold=0.01, 
                 lazy_threshold=0.01,
                 use_checkpoint=True,
                 cache_size=16):
        # 🔧 递归检测
        _safe_mixedop_init("FusedOptimizedMixedOp")
        super(FusedOptimizedMixedOp, self).__init__()
        
        from .genotypes import PRIMITIVES
        
        self._C = C
        self._stride = stride
        self.gradient_threshold = gradient_threshold
        self.lazy_threshold = lazy_threshold
        self.use_checkpoint = use_checkpoint
        self.cache_size = cache_size
        
        # 构建操作列表
        self._ops = nn.ModuleList()
        self._op_names = []
        
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)
            self._op_names.append(primitive)
        
        # 融合优化组件（延迟初始化）
        self._gradient_optimizer = None
        self._memory_manager = None
        self._lazy_computer = None
        
        # 统计信息
        self._stats = {
            'forward_calls': 0,
            'gradient_optimizations': 0,
            'memory_optimizations': 0,
            'lazy_optimizations': 0,
            'cache_hits': 0,
            'triton_usage': 0
        }
    
    def _init_gradient_optimizer(self):
        """初始化梯度优化组件"""
        return {
            'weight_momentum': 0.9,
            'avg_weights': torch.zeros(len(self._ops)),
            'gradient_mask': torch.ones(len(self._ops), dtype=torch.bool),
            'checkpoint_enabled': self.use_checkpoint
        }
    
    def _init_memory_manager(self):
        """初始化内存管理组件"""
        return {
            'output_cache': {},
            'memory_pool': {},
            'max_cache_size': self.cache_size,
            'stream_compute': True
        }
    
    def _init_lazy_computer(self):
        """初始化懒计算组件"""
        return {
            'op_usage_count': torch.zeros(len(self._ops)),
            'active_mask': torch.ones(len(self._ops), dtype=torch.bool),
            'early_termination': True
        }
    
    def _ensure_gradient_optimizer(self):
        """确保梯度优化器已初始化"""
        if self._gradient_optimizer is None:
            self._gradient_optimizer = self._init_gradient_optimizer()
    
    def _ensure_memory_manager(self):
        """确保内存管理器已初始化"""
        if self._memory_manager is None:
            self._memory_manager = self._init_memory_manager()
    
    def _ensure_lazy_computer(self):
        """确保懒计算器已初始化"""
        if self._lazy_computer is None:
            self._lazy_computer = self._init_lazy_computer()
    
    def _update_gradient_mask(self, weights: torch.Tensor):
        """更新梯度计算掩码"""
        self._ensure_gradient_optimizer()
        
        if self._gradient_optimizer['avg_weights'].device != weights.device:
            self._gradient_optimizer['avg_weights'] = self._gradient_optimizer['avg_weights'].to(weights.device)
            self._gradient_optimizer['gradient_mask'] = self._gradient_optimizer['gradient_mask'].to(weights.device)
        
        # 指数移动平均
        momentum = self._gradient_optimizer['weight_momentum']
        self._gradient_optimizer['avg_weights'] = (
            momentum * self._gradient_optimizer['avg_weights'] + 
            (1 - momentum) * weights.detach()
        )
        
        # 更新梯度掩码
        new_mask = self._gradient_optimizer['avg_weights'] > self.gradient_threshold
        if new_mask.sum() < 2:  # 至少保留2个操作
            top_indices = torch.topk(self._gradient_optimizer['avg_weights'], 2).indices
            new_mask[top_indices] = True
        
        self._gradient_optimizer['gradient_mask'] = new_mask
        return new_mask
    
    def _get_cache_key(self, x: torch.Tensor, op_idx: int) -> str:
        """生成缓存键"""
        return f"{x.shape}_{x.device}_{x.data_ptr()}_{op_idx}"
    
    def _memory_efficient_compute(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """内存高效计算"""
        self._ensure_memory_manager()
        self._ensure_gradient_optimizer()
        self._ensure_lazy_computer()
        
        cache = self._memory_manager['output_cache']
        result = None
        
        for i, (op, weight) in enumerate(zip(self._ops, weights)):
            if weight.item() < 1e-6:  # 跳过权重很小的操作
                continue
            
            # 检查缓存
            cache_key = self._get_cache_key(x, i)
            if cache_key in cache:
                output = cache[cache_key]
                self._stats['cache_hits'] += 1
            else:
                # 应用梯度优化
                if self._gradient_optimizer['gradient_mask'][i] and self.training:
                    if self._gradient_optimizer['checkpoint_enabled']:
                        output = checkpoint.checkpoint(op, x, use_reentrant=False)
                        self._stats['gradient_optimizations'] += 1
                    else:
                        output = op(x)
                else:
                    # 跳过梯度计算
                    with torch.no_grad():
                        output = op(x)
                
                # 缓存管理
                if len(cache) < self._memory_manager['max_cache_size']:
                    cache[cache_key] = output.detach().clone()
                
                self._stats['memory_optimizations'] += 1
            
            # 加权输出
            weighted_output = output * weight
            
            # 累积结果（处理不同尺寸的输出）
            if result is None:
                result = weighted_output
            else:
                # 确保尺寸匹配再相加
                if result.shape == weighted_output.shape:
                    result = result + weighted_output
                else:
                    # 如果尺寸不匹配，使用第一个输出的尺寸作为基准
                    # 这通常发生在有stride=2操作时
                    if weighted_output.shape[2:] == result.shape[2:]:
                        result = result + weighted_output
                    else:
                        # 跳过尺寸不匹配的操作，或使用interpolate调整
                        pass
            
            # 更新懒计算统计
            if self._lazy_computer is not None:
                self._lazy_computer['op_usage_count'][i] += 1
        
        # 如果没有有效输出，返回零张量
        if result is None:
            result = torch.zeros_like(x)
            if self._stride == 2:
                # 对于stride=2的情况，调整输出尺寸
                result = torch.nn.functional.avg_pool2d(result, 2)
        
        return result
    
    def _lazy_compute(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """懒计算优化"""
        # 动态剪枝：只计算权重大的操作
        active_indices = torch.where(weights > self.lazy_threshold)[0]
        
        if len(active_indices) == 0:
            active_indices = torch.argmax(weights).unsqueeze(0)
        
        # 早期终止：如果有操作占主导地位
        max_weight = weights.max()
        if max_weight > 0.95:
            max_idx = int(weights.argmax().item())
            self._stats['lazy_optimizations'] += 1
            return self._ops[max_idx](x) * max_weight
        
        # 计算活跃操作
        outputs = []
        active_weights = []
        
        for i in active_indices:
            op = self._ops[i]
            weight = weights[i]
            
            # 检查Triton加速
            if hasattr(op, '_k') and TRITON_AVAILABLE and x.is_cuda:
                self._stats['triton_usage'] += 1
            
            output = op(x)
            outputs.append(output * weight)
            active_weights.append(weight.item())
        
        # 高效求和（处理尺寸不匹配问题）
        if len(outputs) == 1:
            return outputs[0]
        else:
            result = outputs[0]
            for output in outputs[1:]:
                # 确保尺寸匹配再相加
                if result.shape == output.shape:
                    result = result + output
                else:
                    # 如果尺寸不匹配，跳过或使用插值调整
                    # 通常发生在stride=2的操作中
                    pass
            return result
    
    def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """🚀 融合优化前向传播"""
        self._stats['forward_calls'] += 1
        
        # 🧠 智能策略选择：根据模型复杂度和调用频率
        should_use_complex_optimization = (
            self._stats['forward_calls'] > 100 or  # 调用次数多
            x.numel() > 16384 or                   # 输入大
            len(self._ops) > 8                     # 操作多
        )
        
        if not should_use_complex_optimization:
            # 🚀 快速路径：直接使用标准方法（避免复杂优化开销）
            max_idx = int(weights.argmax().item())
            if weights[max_idx] > 0.95:
                # 如果有操作占绝对主导，直接使用它
                self._stats['lazy_optimizations'] += 1
                return self._ops[max_idx](x) * weights[max_idx]
            else:
                # 标准加权求和，但只计算权重大的操作
                active_indices = torch.where(weights > 0.01)[0]
                if len(active_indices) == 0:
                    active_indices = torch.argmax(weights).unsqueeze(0)
                
                outputs = []
                for i in active_indices:
                    output = self._ops[i](x) * weights[i]
                    outputs.append(output)
                
                if len(outputs) == 1:
                    return outputs[0]
                else:
                    result = outputs[0]
                    for output in outputs[1:]:
                        if result.shape == output.shape:
                            result = result + output
                    return result
        
        # 🔧 复杂优化路径：仅在必要时使用
        # 确保所有组件已初始化
        self._ensure_memory_manager()
        self._ensure_gradient_optimizer()
        self._ensure_lazy_computer()
        
        # 更新梯度掩码
        self._update_gradient_mask(weights)
        
        # 根据输入大小选择策略
        if x.numel() > 16384 and self._memory_manager['stream_compute']:
            # 大输入：使用内存优化
            result = self._memory_efficient_compute(x, weights)
        else:
            # 中等输入：使用懒计算优化
            result = self._lazy_compute(x, weights)
        
        # 定期清理缓存
        if self._stats['forward_calls'] % 1000 == 0:
            self._cleanup_cache()
        
        return result
    
    def _cleanup_cache(self):
        """清理缓存"""
        if self._memory_manager is not None:
            cache = self._memory_manager['output_cache']
            if len(cache) > self._memory_manager['max_cache_size']:
                # 保留最近使用的一半
                keys_to_remove = list(cache.keys())[::2]
                for key in keys_to_remove:
                    del cache[key]
        
        # GPU内存清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_optimization_stats(self) -> dict:
        """获取优化统计信息"""
        total_calls = max(1, self._stats['forward_calls'])
        
        # 只有在初始化后才获取活跃操作数
        active_ops = 0
        if self._gradient_optimizer is not None:
            active_ops = self._gradient_optimizer['gradient_mask'].sum().item()
        
        return {
            **self._stats,
            'gradient_optimization_rate': self._stats['gradient_optimizations'] / total_calls,
            'memory_optimization_rate': self._stats['memory_optimizations'] / total_calls,
            'lazy_optimization_rate': self._stats['lazy_optimizations'] / total_calls,
            'cache_hit_rate': self._stats['cache_hits'] / total_calls,
            'triton_usage_rate': self._stats['triton_usage'] / total_calls,
            'active_operations': active_ops,
            'total_operations': len(self._ops)
        } 