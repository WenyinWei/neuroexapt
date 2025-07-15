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
        x_ptr, y_ptr, B, C, H, W, stride: tl.constexpr, BLOCK_C: tl.constexpr
    ):
        pad = 1
        OH = (H + 2 * pad - 3) // stride + 1
        OW = (W + 2 * pad - 3) // stride + 1

        c_block = tl.program_id(0)
        batch = tl.program_id(1)

        offs_c = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C

        x_base = x_ptr + batch * C * H * W + offs_c[:, None, None] * H * W
        y_base = y_ptr + batch * C * OH * OW + offs_c[:, None, None] * OH * OW

        for oh in range(OH):
            ih_base = oh * stride - pad
            for ow in range(OW):
                iw_base = ow * stride - pad
                acc = tl.zeros([BLOCK_C], dtype=tl.float32)
                for kh in tl.static_range(3):
                    ih = ih_base + kh
                    valid_h = (0 <= ih) & (ih < H)
                    for kw in tl.static_range(3):
                        iw = iw_base + kw
                        valid_w = (0 <= iw) & (iw < W)
                        valid = valid_h & valid_w & mask_c
                        x_idx = ih * W + iw
                        x_val = tl.load(x_base + x_idx, mask=valid, other=0.0)
                        acc += x_val
                acc = acc / 9.0
                y_idx = oh * OW + ow
                tl.store(y_base + y_idx, acc, mask=mask_c)

    @triton.jit
    def _maxpool3x3_kernel(
        x_ptr, y_ptr, B, C, H, W, stride: tl.constexpr, BLOCK_C: tl.constexpr
    ):
        pad = 1
        OH = (H + 2 * pad - 3) // stride + 1
        OW = (W + 2 * pad - 3) // stride + 1

        c_block = tl.program_id(0)
        batch = tl.program_id(1)

        offs_c = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C

        x_base = x_ptr + batch * C * H * W + offs_c[:, None, None] * H * W
        y_base = y_ptr + batch * C * OH * OW + offs_c[:, None, None] * OH * OW

        for oh in range(OH):
            ih_base = oh * stride - pad
            for ow in range(OW):
                iw_base = ow * stride - pad
                acc = tl.full([BLOCK_C], -3.4e38, dtype=tl.float32)
                for kh in tl.static_range(3):
                    ih = ih_base + kh
                    valid_h = (0 <= ih) & (ih < H)
                    for kw in tl.static_range(3):
                        iw = iw_base + kw
                        valid_w = (0 <= iw) & (iw < W)
                        valid = valid_h & valid_w & mask_c
                        x_idx = ih * W + iw
                        x_val = tl.load(x_base + x_idx, mask=valid, other=-3.4e38)
                        acc = tl.maximum(acc, x_val)
                y_idx = oh * OW + ow
                tl.store(y_base + y_idx, acc, mask=mask_c)

    @triton.jit
    def _pool_generic_kernel(
        x_ptr, y_ptr, B, C, H, W,
        K: tl.constexpr,
        stride: tl.constexpr,
        is_max: tl.constexpr,
        BLOCK_C: tl.constexpr
    ):
        """Generic pool kernel for any K size."""
        pad = K // 2
        OH = (H + 2 * pad - K) // stride + 1
        OW = (W + 2 * pad - K) // stride + 1

        c_block = tl.program_id(0)
        batch = tl.program_id(1)

        offs_c = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C

        x_base = x_ptr + batch * C * H * W + offs_c[:, None, None] * H * W
        y_base = y_ptr + batch * C * OH * OW + offs_c[:, None, None] * OH * OW

        for oh in range(OH):
            ih_base = oh * stride - pad
            for ow in range(OW):
                iw_base = ow * stride - pad
                if is_max:
                    acc = tl.full([BLOCK_C], -3.4e38, dtype=tl.float32)
                else:
                    acc = tl.zeros([BLOCK_C], dtype=tl.float32)
                
                for kh in tl.static_range(K):
                    ih = ih_base + kh
                    valid_h = (0 <= ih) & (ih < H)
                    for kw in tl.static_range(K):
                        iw = iw_base + kw
                        valid_w = (0 <= iw) & (iw < W)
                        valid = valid_h & valid_w & mask_c
                        x_idx = ih * W + iw
                        if is_max:
                            x_val = tl.load(x_base + x_idx, mask=valid, other=-3.4e38)
                            acc = tl.maximum(acc, x_val)
                        else:
                            x_val = tl.load(x_base + x_idx, mask=valid, other=0.0)
                            acc += x_val
                
                if not is_max:
                    acc = acc / (K * K)
                y_idx = oh * OW + ow
                tl.store(y_base + y_idx, acc, mask=mask_c)

    @triton.jit
    def _global_avgpool_kernel(
        x_ptr, y_ptr, B, C, H, W, BLOCK_C: tl.constexpr
    ):
        """Global average pooling: [B,C,H,W] -> [B,C,1,1]"""
        c_block = tl.program_id(0)
        batch = tl.program_id(1)

        offs_c = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C

        x_base = x_ptr + batch * C * H * W + offs_c[:, None, None] * H * W
        y_base = y_ptr + batch * C + offs_c

        acc = tl.zeros([BLOCK_C], dtype=tl.float32)
        for h in range(H):
            for w in range(W):
                x_idx = h * W + w
                x_val = tl.load(x_base + x_idx, mask=mask_c, other=0.0)
                acc += x_val
        
        acc = acc / (H * W)
        tl.store(y_base, acc, mask=mask_c)

    def _pool3x3_forward(x: torch.Tensor, is_max: bool, stride: int = 1):
        B, C, H, W = x.shape
        OH = (H + 2 - 3) // stride + 1
        OW = (W + 2 - 3) // stride + 1
        y = torch.empty((B, C, OH, OW), device=x.device, dtype=x.dtype)
        BLOCK_C = 32
        grid = (triton.cdiv(C, BLOCK_C), B)
        kernel = _maxpool3x3_kernel if is_max else _avgpool3x3_kernel
        kernel[grid](x, y, B, C, H, W, stride, BLOCK_C=BLOCK_C, num_warps=4)  # type: ignore[arg-type]
        return y

    def avg_pool3x3_forward(x: torch.Tensor, stride: int = 1):
        return _pool3x3_forward(x, False, stride)

    def max_pool3x3_forward(x: torch.Tensor, stride: int = 1):
        return _pool3x3_forward(x, True, stride)

    def _pool_generic_forward(x: torch.Tensor, kernel_size: int, stride: int, is_max: bool):
        B, C, H, W = x.shape
        pad = kernel_size // 2
        OH = (H + 2 * pad - kernel_size) // stride + 1
        OW = (W + 2 * pad - kernel_size) // stride + 1
        y = torch.empty((B, C, OH, OW), device=x.device, dtype=x.dtype)
        BLOCK_C = 32
        grid = (triton.cdiv(C, BLOCK_C), B)
        _pool_generic_kernel[grid](  # type: ignore[arg-type]
            x, y, B, C, H, W, kernel_size, stride, is_max, BLOCK_C=BLOCK_C, num_warps=4
        )
        return y

    def avg_pool5x5_forward(x: torch.Tensor, stride: int = 1):
        return _pool_generic_forward(x, 5, stride, False)

    def max_pool5x5_forward(x: torch.Tensor, stride: int = 1):
        return _pool_generic_forward(x, 5, stride, True)

    def avg_pool7x7_forward(x: torch.Tensor, stride: int = 1):
        return _pool_generic_forward(x, 7, stride, False)

    def max_pool7x7_forward(x: torch.Tensor, stride: int = 1):
        return _pool_generic_forward(x, 7, stride, True)

    def global_avgpool_forward(x: torch.Tensor):
        B, C, H, W = x.shape
        y = torch.empty((B, C, 1, 1), device=x.device, dtype=x.dtype)
        BLOCK_C = 32
        grid = (triton.cdiv(C, BLOCK_C), B)
        _global_avgpool_kernel[grid](  # type: ignore[arg-type]
            x, y, B, C, H, W, BLOCK_C=BLOCK_C, num_warps=4
        )
        return y

else:

    def avg_pool3x3_forward(x: torch.Tensor, stride: int = 1):  # type: ignore
        return torch.nn.functional.avg_pool2d(x, 3, stride=stride, padding=1, count_include_pad=False)

    def max_pool3x3_forward(x: torch.Tensor, stride: int = 1):  # type: ignore
        return torch.nn.functional.max_pool2d(x, 3, stride=stride, padding=1)

    def avg_pool5x5_forward(x: torch.Tensor, stride: int = 1):  # type: ignore
        return torch.nn.functional.avg_pool2d(x, 5, stride=stride, padding=2, count_include_pad=False)

    def max_pool5x5_forward(x: torch.Tensor, stride: int = 1):  # type: ignore
        return torch.nn.functional.max_pool2d(x, 5, stride=stride, padding=2)

    def avg_pool7x7_forward(x: torch.Tensor, stride: int = 1):  # type: ignore
        return torch.nn.functional.avg_pool2d(x, 7, stride=stride, padding=3, count_include_pad=False)

    def max_pool7x7_forward(x: torch.Tensor, stride: int = 1):  # type: ignore
        return torch.nn.functional.max_pool2d(x, 7, stride=stride, padding=3)

    def global_avgpool_forward(x: torch.Tensor):  # type: ignore
        return torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))

    TRITON_AVAILABLE = False 