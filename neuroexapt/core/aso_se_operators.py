"""
ASO-SE 神经架构搜索操作符框架
重新设计的稳定架构搜索基础设施
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class StableOp(nn.Module):
    """稳定的基础操作类"""
    
    def __init__(self, C, stride, affine=True):
        super().__init__()
        self.C = C
        self.stride = stride
        self.affine = affine
    
    def forward(self, x):
        raise NotImplementedError
    
    def get_flops(self, input_shape):
        """计算FLOPS，用于效率评估"""
        return 0


class Identity(StableOp):
    """恒等映射"""
    
    def forward(self, x):
        if self.stride == 1:
            return x
        else:
            # 下采样的恒等映射
            return x[:, :, ::self.stride, ::self.stride]


class Zero(StableOp):
    """零操作"""
    
    def forward(self, x):
        if self.stride == 1:
            return torch.zeros_like(x)
        else:
            shape = list(x.shape)
            shape[2] = (shape[2] + self.stride - 1) // self.stride
            shape[3] = (shape[3] + self.stride - 1) // self.stride
            return torch.zeros(shape, dtype=x.dtype, device=x.device)


class ReLUConvBN(StableOp):
    """ReLU + Conv + BatchNorm"""
    
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__(C_out, stride, affine)
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )
    
    def forward(self, x):
        return self.op(x)


class SepConv(StableOp):
    """深度可分离卷积"""
    
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__(C_out, stride, affine)
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, 
                     padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, 
                     padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )
    
    def forward(self, x):
        return self.op(x)


class DilConv(StableOp):
    """扩张卷积"""
    
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super().__init__(C_out, stride, affine)
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, 
                     padding=padding, dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )
    
    def forward(self, x):
        return self.op(x)


class FactorizedReduce(StableOp):
    """因式化降维"""
    
    def __init__(self, C_in, C_out, affine=True):
        super().__init__(C_out, 2, affine)
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False) 
        self.bn = nn.BatchNorm2d(C_out, affine=affine)
    
    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
        out = self.bn(out)
        return out


# 定义操作符映射
PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3', 
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
    'conv_1x1',
    'conv_3x3',
]


def create_operation(primitive, C_in, C_out, stride, affine=True):
    """创建操作实例"""
    if primitive == 'none':
        return Zero(C_out, stride, affine)
    elif primitive == 'max_pool_3x3':
        return nn.Sequential(
            nn.MaxPool2d(3, stride=stride, padding=1),
            nn.BatchNorm2d(C_in, affine=affine) if C_in == C_out else 
            nn.Conv2d(C_in, C_out, 1, bias=False)
        )
    elif primitive == 'avg_pool_3x3':
        return nn.Sequential(
            nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
            nn.BatchNorm2d(C_in, affine=affine) if C_in == C_out else 
            nn.Conv2d(C_in, C_out, 1, bias=False)
        )
    elif primitive == 'skip_connect':
        if stride == 1 and C_in == C_out:
            return Identity(C_out, stride, affine)
        else:
            return FactorizedReduce(C_in, C_out, affine)
    elif primitive == 'sep_conv_3x3':
        return SepConv(C_in, C_out, 3, stride, 1, affine)
    elif primitive == 'sep_conv_5x5':
        return SepConv(C_in, C_out, 5, stride, 2, affine)
    elif primitive == 'dil_conv_3x3':
        return DilConv(C_in, C_out, 3, stride, 2, 2, affine)
    elif primitive == 'dil_conv_5x5':
        return DilConv(C_in, C_out, 5, stride, 4, 2, affine)
    elif primitive == 'conv_1x1':
        return ReLUConvBN(C_in, C_out, 1, stride, 0, affine)
    elif primitive == 'conv_3x3':
        return ReLUConvBN(C_in, C_out, 3, stride, 1, affine)
    else:
        raise ValueError(f"Unknown primitive: {primitive}")


class StableMixedOp(nn.Module):
    """稳定的混合操作"""
    
    def __init__(self, C_in, C_out, stride, primitives=None):
        super().__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride
        
        if primitives is None:
            primitives = PRIMITIVES
        
        self.primitives = primitives
        self.operations = nn.ModuleList()
        
        # 创建所有候选操作
        for primitive in primitives:
            op = create_operation(primitive, C_in, C_out, stride)
            self.operations.append(op)
        
        # 预计算操作权重，避免运行时计算
        self._cached_weights = None
        self._cache_valid = False
        
    def forward(self, x, weights):
        """前向传播，使用稳定的加权求和"""
        # 输入验证
        if torch.isnan(weights).any() or torch.isinf(weights).any():
            # 安全回退：使用skip_connect
            skip_idx = self.primitives.index('skip_connect') if 'skip_connect' in self.primitives else 0
            return self.operations[skip_idx](x)
        
        # 确保权重和为1
        weights = F.softmax(weights, dim=0)
        
        # 智能操作选择策略
        max_weight_idx = torch.argmax(weights).item()
        max_weight = weights[max_weight_idx].item()
        
        # 如果有明显的主导操作 (>0.7)，主要使用该操作
        if max_weight > 0.7:
            dominant_result = self.operations[max_weight_idx](x)
            
            # 如果不是完全确定，加入少量其他操作的贡献
            if max_weight < 0.95:
                other_result = 0.0
                other_weight_sum = 0.0
                
                for i, op in enumerate(self.operations):
                    if i != max_weight_idx and weights[i] > 0.05:
                        try:
                            other_result += weights[i] * op(x)
                            other_weight_sum += weights[i]
                        except Exception:
                            continue
                
                if other_weight_sum > 0:
                    # 重新归一化
                    total_weight = max_weight + other_weight_sum
                    return (max_weight / total_weight) * dominant_result + \
                           (other_weight_sum / total_weight) * other_result
            
            return dominant_result
        
        # 如果没有明显主导操作，计算加权和
        result = 0.0
        total_weight = 0.0
        
        for i, op in enumerate(self.operations):
            weight = weights[i]
            if weight > 0.01:  # 忽略权重太小的操作
                try:
                    op_result = op(x)
                    result += weight * op_result
                    total_weight += weight
                except Exception as e:
                    print(f"Warning: Operation {i} ({self.primitives[i]}) failed: {e}")
                    continue
        
        if total_weight < 0.1:
            # 如果所有操作都失败，使用skip连接
            skip_idx = self.primitives.index('skip_connect') if 'skip_connect' in self.primitives else 0
            return self.operations[skip_idx](x)
        
        return result / total_weight if total_weight > 0 else result
    
    def get_flops(self, input_shape, weights):
        """计算加权FLOPS"""
        total_flops = 0
        for i, op in enumerate(self.operations):
            if hasattr(op, 'get_flops'):
                total_flops += weights[i] * op.get_flops(input_shape)
        return total_flops