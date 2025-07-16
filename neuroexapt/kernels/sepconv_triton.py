import torch
from typing import Optional

# Try to import Triton. If unavailable, we silently fall back to the PyTorch implementation.
try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore
    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover – execution environment may miss triton
    TRITON_AVAILABLE = False

######################################################################
#                           Triton kernel                            #
######################################################################

if TRITON_AVAILABLE:

    @triton.jit
    def _dwconv3x3_kernel(
        x_ptr,  # [B, C, H, W]
        w_ptr,  # [C, 1, 3, 3]
        y_ptr,
        B, C, H, W,
        stride: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Depth-wise 3×3 conv (padding=1) with optional stride."""
        pid = tl.program_id(0)
        
        OH = (H + 2 - 3) // stride + 1
        OW = (W + 2 - 3) // stride + 1
        total_outputs = B * C * OH * OW
        
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_outputs
        
        # 将1D索引转换为4D坐标 (b, c, oh, ow)
        ow = offsets % OW
        oh = (offsets // OW) % OH
        c = (offsets // (OW * OH)) % C
        b = offsets // (OW * OH * C)
        
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        
        # 3x3深度卷积
        for kh in tl.static_range(3):
            for kw in tl.static_range(3):
                ih = oh * stride + kh - 1  # padding=1
                iw = ow * stride + kw - 1
                
                # 边界检查
                valid = mask & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                
                # 输入索引
                input_idx = b * C * H * W + c * H * W + ih * W + iw
                # 权重索引 (每个通道有自己的3x3权重)
                weight_idx = c * 9 + kh * 3 + kw  # 9 = 3*3
                
                x_val = tl.load(x_ptr + input_idx, mask=valid, other=0.0)
                w_val = tl.load(w_ptr + weight_idx, mask=(c < C), other=0.0)
                
                acc += x_val * w_val
        
        # 存储结果
        output_idx = b * C * OH * OW + c * OH * OW + oh * OW + ow
        tl.store(y_ptr + output_idx, acc, mask=mask)

    @triton.jit  
    def _pointwise_conv_kernel(
        x_ptr,  # [B, C_in, H, W]
        w_ptr,  # [C_out, C_in, 1, 1]
        bias_ptr,  # [C_out]
        y_ptr,  # [B, C_out, H, W]
        B, C_in, C_out, H, W,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Point-wise 1×1 conv."""
        pid = tl.program_id(0)
        
        total_outputs = B * C_out * H * W
        
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_outputs
        
        # 将1D索引转换为4D坐标 (b, c_out, h, w)
        w_idx = offsets % W
        h_idx = (offsets // W) % H
        c_out = (offsets // (W * H)) % C_out
        b = offsets // (W * H * C_out)
        
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        
        # 点卷积：对所有输入通道求和
        for c_in in range(C_in):
            input_idx = b * C_in * H * W + c_in * H * W + h_idx * W + w_idx
            weight_idx = c_out * C_in + c_in  # 1x1卷积权重
            
            x_val = tl.load(x_ptr + input_idx, mask=mask, other=0.0)
            w_val = tl.load(w_ptr + weight_idx, mask=(c_out < C_out), other=0.0)
            
            acc += x_val * w_val
        
        # 加bias
        if bias_ptr is not None:
            bias_val = tl.load(bias_ptr + c_out, mask=(c_out < C_out), other=0.0)
            acc += bias_val
        
        # 存储结果
        output_idx = b * C_out * H * W + c_out * H * W + h_idx * W + w_idx
        tl.store(y_ptr + output_idx, acc, mask=mask)

######################################################################
#                    Separable Convolution Functions                 #
######################################################################

def sepconv_forward_generic(
    x: torch.Tensor,
    dw_weight: torch.Tensor,
    pw_weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    kernel_size: int = 3,
    stride: int = 1,
    dilation: int = 1,
) -> torch.Tensor:
    """Generic separable convolution: depth-wise + point-wise."""
    
    # 如果Triton不可用或参数不支持，使用PyTorch fallback
    if not TRITON_AVAILABLE or kernel_size != 3 or stride != 1 or dilation != 1:
        # PyTorch fallback implementation
        y = torch.nn.functional.conv2d(
            x, dw_weight, bias=None, stride=stride, 
            padding=((kernel_size - 1) * dilation) // 2, 
            dilation=dilation, groups=x.size(1)
        )
        y = torch.nn.functional.conv2d(y, pw_weight, bias=bias)
        return y
    
    B, C, H, W = x.shape
    C_out = pw_weight.size(0)
    
    # 第一步: 深度卷积
    OH = (H + 2 - 3) // stride + 1
    OW = (W + 2 - 3) // stride + 1
    dw_out = torch.empty(B, C, OH, OW, device=x.device, dtype=x.dtype)
    
    total_outputs = B * C * OH * OW
    BLOCK_SIZE = 256
    grid = (triton.cdiv(total_outputs, BLOCK_SIZE),)
    
    _dwconv3x3_kernel[grid](
        x, dw_weight, dw_out, B, C, H, W, stride, BLOCK_SIZE
    )
    
    # 第二步: 点卷积
    pw_out = torch.empty(B, C_out, OH, OW, device=x.device, dtype=x.dtype)
    
    total_outputs = B * C_out * OH * OW
    grid = (triton.cdiv(total_outputs, BLOCK_SIZE),)
    
    _pointwise_conv_kernel[grid](
        dw_out, pw_weight, bias, pw_out, B, C, C_out, OH, OW, BLOCK_SIZE
    )
    
    return pw_out

# 为了保持向后兼容，提供一些特化版本
def sepconv3x3_forward(
    x: torch.Tensor,
    dw_weight: torch.Tensor,
    pw_weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: int = 1,
) -> torch.Tensor:
    """3x3 separable convolution."""
    return sepconv_forward_generic(x, dw_weight, pw_weight, bias, 3, stride, 1)

def sepconv5x5_forward(
    x: torch.Tensor,
    dw_weight: torch.Tensor,
    pw_weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: int = 1,
) -> torch.Tensor:
    """5x5 separable convolution (fallback to PyTorch)."""
    return sepconv_forward_generic(x, dw_weight, pw_weight, bias, 5, stride, 1)

def sepconv7x7_forward(
    x: torch.Tensor,
    dw_weight: torch.Tensor, 
    pw_weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: int = 1,
) -> torch.Tensor:
    """7x7 separable convolution (fallback to PyTorch)."""
    return sepconv_forward_generic(x, dw_weight, pw_weight, bias, 7, stride, 1)

# 为了向后兼容，提供原始的函数名
def sepconv_forward(
    x: torch.Tensor,
    dw_weight: torch.Tensor,
    pw_weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    kernel_size: int = 3,
    stride: int = 1,
    dilation: int = 1,
) -> torch.Tensor:
    """Backward compatibility alias for sepconv_forward_generic."""
    return sepconv_forward_generic(x, dw_weight, pw_weight, bias, kernel_size, stride, dilation) 