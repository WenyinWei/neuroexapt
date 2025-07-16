import torch
from typing import Optional

try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore
    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    TRITON_AVAILABLE = False

__all__ = [
    "avg_pool3x3_forward", "max_pool3x3_forward",
    "avg_pool5x5_forward", "max_pool5x5_forward", 
    "avg_pool7x7_forward", "max_pool7x7_forward",
    "global_avgpool_forward", "TRITON_AVAILABLE"
]

if TRITON_AVAILABLE:

    @triton.jit
    def _avgpool3x3_kernel(
        x_ptr, y_ptr, 
        B, C, H, W, 
        stride: tl.constexpr, 
        BLOCK_SIZE: tl.constexpr
    ):
        # 获取程序ID
        pid = tl.program_id(0)
        
        # 计算输出尺寸
        pad = 1
        OH = (H + 2 * pad - 3) // stride + 1
        OW = (W + 2 * pad - 3) // stride + 1
        
        # 计算当前线程负责的输出位置
        total_outputs = B * C * OH * OW
        
        # 每个线程处理BLOCK_SIZE个元素
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_outputs
        
        # 将1D索引转换为4D坐标 (b, c, oh, ow)
        ow = offsets % OW
        oh = (offsets // OW) % OH
        c = (offsets // (OW * OH)) % C
        b = offsets // (OW * OH * C)
        
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        
        # 3x3池化窗口
        for kh in tl.static_range(3):
            for kw in tl.static_range(3):
                ih = oh * stride + kh - pad
                iw = ow * stride + kw - pad
                
                # 边界检查
                valid = mask & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                
                # 计算输入索引
                input_idx = b * C * H * W + c * H * W + ih * W + iw
                
                # 加载输入值 - 边界外为0（模拟padding）
                x_val = tl.load(x_ptr + input_idx, mask=valid, other=0.0)
                acc += x_val
        
        # 计算平均值 - PyTorch默认count_include_pad=True，所以分母总是9
        result = acc / 9.0
        
        # 存储结果
        output_idx = b * C * OH * OW + c * OH * OW + oh * OW + ow
        tl.store(y_ptr + output_idx, result, mask=mask)

    @triton.jit
    def _maxpool3x3_kernel(
        x_ptr, y_ptr, 
        B, C, H, W, 
        stride: tl.constexpr, 
        BLOCK_SIZE: tl.constexpr
    ):
        # 获取程序ID
        pid = tl.program_id(0)
        
        # 计算输出尺寸
        pad = 1
        OH = (H + 2 * pad - 3) // stride + 1
        OW = (W + 2 * pad - 3) // stride + 1
        
        # 计算当前线程负责的输出位置
        total_outputs = B * C * OH * OW
        
        # 每个线程处理BLOCK_SIZE个元素
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_outputs
        
        # 将1D索引转换为4D坐标 (b, c, oh, ow)
        ow = offsets % OW
        oh = (offsets // OW) % OH
        c = (offsets // (OW * OH)) % C
        b = offsets // (OW * OH * C)
        
        max_val = tl.full((BLOCK_SIZE,), -float('inf'), dtype=tl.float32)
        
        # 3x3池化窗口
        for kh in tl.static_range(3):
            for kw in tl.static_range(3):
                ih = oh * stride + kh - pad
                iw = ow * stride + kw - pad
                
                # 边界检查
                valid = mask & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                
                # 计算输入索引
                input_idx = b * C * H * W + c * H * W + ih * W + iw
                
                # 加载输入值
                x_val = tl.load(x_ptr + input_idx, mask=valid, other=-float('inf'))
                max_val = tl.maximum(max_val, x_val)
        
        # 存储结果
        output_idx = b * C * OH * OW + c * OH * OW + oh * OW + ow
        tl.store(y_ptr + output_idx, max_val, mask=mask)

######################################################################
#                      Pool Function Wrappers                       #
######################################################################

def avg_pool3x3_forward(x: torch.Tensor, stride: int = 1) -> torch.Tensor:
    """3x3 average pooling with padding=1."""
    if not TRITON_AVAILABLE:
        return torch.nn.functional.avg_pool2d(x, 3, stride=stride, padding=1)
    
    B, C, H, W = x.shape
    pad = 1
    OH = (H + 2 * pad - 3) // stride + 1
    OW = (W + 2 * pad - 3) // stride + 1
    
    y = torch.empty(B, C, OH, OW, device=x.device, dtype=x.dtype)
    
    total_outputs = B * C * OH * OW
    BLOCK_SIZE = 256
    grid = (triton.cdiv(total_outputs, BLOCK_SIZE),)
    
    _avgpool3x3_kernel[grid](
        x, y, B, C, H, W, stride, BLOCK_SIZE
    )
    
    return y

def max_pool3x3_forward(x: torch.Tensor, stride: int = 1) -> torch.Tensor:
    """3x3 max pooling with padding=1."""
    if not TRITON_AVAILABLE:
        return torch.nn.functional.max_pool2d(x, 3, stride=stride, padding=1)
    
    B, C, H, W = x.shape
    pad = 1
    OH = (H + 2 * pad - 3) // stride + 1
    OW = (W + 2 * pad - 3) // stride + 1
    
    y = torch.empty(B, C, OH, OW, device=x.device, dtype=x.dtype)
    
    total_outputs = B * C * OH * OW
    BLOCK_SIZE = 256
    grid = (triton.cdiv(total_outputs, BLOCK_SIZE),)
    
    _maxpool3x3_kernel[grid](
        x, y, B, C, H, W, stride, BLOCK_SIZE
    )
    
    return y

# 其他池化函数的简化实现（使用PyTorch fallback）
def avg_pool5x5_forward(x: torch.Tensor, stride: int = 1) -> torch.Tensor:
    """5x5 average pooling with padding=2."""
    return torch.nn.functional.avg_pool2d(x, 5, stride=stride, padding=2)

def max_pool5x5_forward(x: torch.Tensor, stride: int = 1) -> torch.Tensor:
    """5x5 max pooling with padding=2."""
    return torch.nn.functional.max_pool2d(x, 5, stride=stride, padding=2)

def avg_pool7x7_forward(x: torch.Tensor, stride: int = 1) -> torch.Tensor:
    """7x7 average pooling with padding=3."""
    return torch.nn.functional.avg_pool2d(x, 7, stride=stride, padding=3)

def max_pool7x7_forward(x: torch.Tensor, stride: int = 1) -> torch.Tensor:
    """7x7 max pooling with padding=3."""
    return torch.nn.functional.max_pool2d(x, 7, stride=stride, padding=3)

def global_avgpool_forward(x: torch.Tensor) -> torch.Tensor:
    """Global average pooling."""
    return torch.nn.functional.adaptive_avg_pool2d(x, 1) 