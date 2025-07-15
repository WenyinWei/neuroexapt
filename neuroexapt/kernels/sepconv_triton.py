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
        BLOCK_C: tl.constexpr,
    ):
        """Depth-wise 3×3 conv (padding=1) with optional stride 2."""
        c_block = tl.program_id(0)
        batch = tl.program_id(1)

        offs_c = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C

        # Pointers offset
        x_ptr = x_ptr + batch * C * H * W + offs_c[:, None, None] * H * W
        w_ptr = w_ptr + offs_c[:, None, None] * 9  # 3*3 kernel size = 9

        OH = (H + 2 - 3) // stride + 1
        OW = (W + 2 - 3) // stride + 1

        for oh in range(OH):
            ih_base = oh * stride - 1
            for ow in range(OW):
                iw_base = ow * stride - 1

                acc = tl.zeros([BLOCK_C], dtype=tl.float32)
                for kh in range(3):
                    ih = ih_base + kh
                    valid_h = (0 <= ih) & (ih < H)
                    for kw in range(3):
                        iw = iw_base + kw
                        valid_w = (0 <= iw) & (iw < W)
                        valid = valid_h & valid_w & mask_c

                        x_idx = ih * W + iw
                        w_idx = kh * 3 + kw

                        x_val = tl.load(x_ptr + x_idx, mask=valid, other=0.0)
                        w_val = tl.load(w_ptr + w_idx, mask=mask_c, other=0.0)
                        acc += x_val * w_val

                # Store result
                y_ptr_base = y_ptr + batch * C * OH * OW + offs_c * OH * OW + oh * OW + ow
                tl.store(y_ptr_base, acc, mask=mask_c)

    ######################################################################
    # Generic KxK depth-wise kernel (K in {3,5,7})                       #
    ######################################################################

    @triton.jit
    def _dwconv_generic_kernel(
        x_ptr,
        w_ptr,
        y_ptr,
        B, C, H, W,
        K: tl.constexpr,
        stride: tl.constexpr,
        dilation: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        pad = ((K - 1) * dilation) // 2

        c_block = tl.program_id(0)
        batch = tl.program_id(1)

        offs_c = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C

        x_ptr = x_ptr + batch * C * H * W + offs_c[:, None, None] * H * W
        w_ptr = w_ptr + offs_c[:, None, None] * (K * K)

        OH = (H + 2 * pad - dilation * (K - 1) - 1) // stride + 1
        OW = (W + 2 * pad - dilation * (K - 1) - 1) // stride + 1

        for oh in range(OH):
            ih_base = oh * stride - pad
            for ow in range(OW):
                iw_base = ow * stride - pad

                acc = tl.zeros([BLOCK_C], dtype=tl.float32)

                for kh in tl.static_range(K):
                    ih = ih_base + kh * dilation
                    valid_h = (0 <= ih) & (ih < H)
                    for kw in tl.static_range(K):
                        iw = iw_base + kw * dilation
                        valid_w = (0 <= iw) & (iw < W)
                        valid = valid_h & valid_w & mask_c

                        x_idx = ih * W + iw
                        w_idx = kh * K + kw

                        x_val = tl.load(x_ptr + x_idx, mask=valid, other=0.0)
                        w_val = tl.load(w_ptr + w_idx, mask=mask_c, other=0.0)
                        acc += x_val * w_val

                y_ptr_base = y_ptr + batch * C * OH * OW + offs_c * OH * OW + oh * OW + ow
                tl.store(y_ptr_base, acc, mask=mask_c)

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
        """Fused separable conv using Triton depth-wise kernel + PyTorch pointwise."""
        if x.is_cpu or dilation not in {1, 2}:
            # Fallback on CPU regardless of Triton availability.
            pad = ((kernel_size - 1) * dilation) // 2
            y = torch.nn.functional.conv2d(
                x,
                dw_weight,
                None,
                stride=stride,
                padding=pad,
                dilation=dilation,
                groups=x.shape[1],
            )
            y = torch.nn.functional.conv2d(y, pw_weight, bias, stride=1, padding=0)
            return y

        B, C, H, W = x.shape
        # compute output size for generic K,dil
        pad = ((kernel_size - 1) * dilation) // 2
        OH = (H + 2 * pad - dilation * (kernel_size - 1) - 1) // stride + 1
        OW = (W + 2 * pad - dilation * (kernel_size - 1) - 1) // stride + 1

        y_dw = torch.empty((B, C, OH, OW), device=x.device, dtype=x.dtype)

        # Choose kernel according to size
        if kernel_size in {3, 5, 7} and dilation in {1, 2}:
            BLOCK_C = 32
            grid = (triton.cdiv(C, BLOCK_C), B)
            if kernel_size == 3 and dilation == 1:
                _dwconv3x3_kernel[grid](  # type: ignore[arg-type]
                    x,
                    dw_weight,
                    y_dw,
                    B,
                    C,
                    H,
                    W,
                    stride,
                    BLOCK_C=BLOCK_C,
                    num_warps=4,
                )
            else:
                _dwconv_generic_kernel[grid](  # type: ignore[arg-type]
                    x,
                    dw_weight,
                    y_dw,
                    B,
                    C,
                    H,
                    W,
                    K=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    BLOCK_C=BLOCK_C,
                    num_warps=4,
                )
        else:
            # Fallback to PyTorch for unsupported kernel sizes currently
            pad = ((kernel_size - 1) * dilation) // 2
            y_dw = torch.nn.functional.conv2d(
                x,
                dw_weight,
                None,
                stride=stride,
                padding=pad,
                dilation=dilation,
                groups=C,
            )

        # Pointwise conv (1×1) – negligible cost compared to depth-wise
        y_out = torch.nn.functional.conv2d(y_dw, pw_weight, bias, stride=1, padding=0)
        return y_out

# ----------------------- Fallback implementation --------------------
else:

    def _sepconv_forward_fallback(
        x: torch.Tensor,
        dw_weight: torch.Tensor,
        pw_weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
    ) -> torch.Tensor:
        """Fallback PyTorch implementation when Triton is unavailable."""
        y = torch.nn.functional.conv2d(
            x,
            dw_weight,
            None,
            stride=stride,
            padding=((kernel_size - 1) * dilation) // 2,
            dilation=dilation,
            groups=x.shape[1],
        )
        y = torch.nn.functional.conv2d(y, pw_weight, bias, stride=1, padding=0)
        return y

    # expose under common name
    sepconv_forward = _sepconv_forward_fallback  # type: ignore[assignment] 