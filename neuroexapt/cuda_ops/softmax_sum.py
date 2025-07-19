"""
defgroup group_softmax_sum Softmax Sum
ingroup core
Softmax Sum module for NeuroExapt framework.
"""


import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline
from typing import Any

# Build only once and cache
_module = None

_cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Forward: out[b,c,h,w] = sum_i w[i]*x[i,b,c,h,w]
// shapes: x [N, B, C, H, W] contiguous, w[N]

template <typename scalar_t>
__global__ void softmax_sum_fwd(const scalar_t* __restrict__ x,
                                const scalar_t* __restrict__ w,
                                scalar_t* __restrict__ out,
                                int N, int B, int C, int H, int W)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * H * W;
    if (idx >= total) return;

    // decode idx to (b,c,h,w)
    int w_idx = idx % W;          // w
    int h_idx = (idx / W) % H;    // h
    int c_idx = (idx / (W*H)) % C;// c
    int b_idx = idx / (W*H*C);    // b

    scalar_t acc = 0;
    for(int n=0; n<N; ++n){
        // linear index into x
        size_t offset = (((size_t)n*B + b_idx)*C + c_idx)*H*W + h_idx*W + w_idx;
        acc += w[n] * x[offset];
    }
    out[idx] = acc;
}

// Backward wrt x and w

template <typename scalar_t>
__global__ void softmax_sum_bwd(const scalar_t* __restrict__ grad_out,
                                const scalar_t* __restrict__ x,
                                const scalar_t* __restrict__ w,
                                scalar_t* __restrict__ grad_x,
                                scalar_t* __restrict__ grad_w,
                                int N, int B, int C, int H, int W)
{
    // compute grad_x in parallel over (n,rest)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * B * C * H * W;
    if (idx >= total) return;

    int w_pos = idx % W;
    int h_pos = (idx / W) % H;
    int c_pos = (idx / (W*H)) % C;
    int b_pos = (idx / (W*H*C)) % B;
    int n_pos = idx / (W*H*C*B);

    size_t off = (((size_t)n_pos*B + b_pos)*C + c_pos)*H*W + h_pos*W + w_pos;

    scalar_t go = grad_out[((size_t)b_pos*C + c_pos)*H*W + h_pos*W + w_pos];
    grad_x[off] = w[n_pos] * go;

    // atomic add for grad_w
    atomicAdd(grad_w + n_pos, x[off] * go);
}

std::vector<torch::Tensor> softmax_sum_forward(torch::Tensor x, torch::Tensor logits){
    const int64_t N = x.size(0);
    auto weights = torch::softmax(logits, 0);
    auto out = torch::empty({x.size(1), x.size(2), x.size(3), x.size(4)}, x.options());

    const int64_t total = x.size(1)*x.size(2)*x.size(3)*x.size(4);
    const int threads = 256;
    const int blocks = (total + threads -1)/threads;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "softmax_sum_fwd", ([&]{
        softmax_sum_fwd<scalar_t><<<blocks, threads>>>(x.data_ptr<scalar_t>(),
                                                      weights.data_ptr<scalar_t>(),
                                                      out.data_ptr<scalar_t>(),
                                                      N, x.size(1), x.size(2), x.size(3), x.size(4));
    }));
    return {out, weights};
}

std::vector<torch::Tensor> softmax_sum_backward(torch::Tensor grad_out,
                                               torch::Tensor x,
                                               torch::Tensor weights){
    const int64_t N = x.size(0);
    auto grad_x = torch::empty_like(x);
    auto grad_logits = torch::zeros({N}, x.options());

    const int64_t total = N* x.size(1)*x.size(2)*x.size(3)*x.size(4);
    const int threads = 256;
    const int blocks = (total + threads -1)/threads;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "softmax_sum_bwd", ([&]{
        softmax_sum_bwd<scalar_t><<<blocks, threads>>>(grad_out.data_ptr<scalar_t>(),
                                                      x.data_ptr<scalar_t>(),
                                                      weights.data_ptr<scalar_t>(),
                                                      grad_x.data_ptr<scalar_t>(),
                                                      grad_logits.data_ptr<scalar_t>(),
                                                      N, x.size(1), x.size(2), x.size(3), x.size(4));
    }));
    return {grad_x, grad_logits};
}
"""

_cpp_src = r"""
std::vector<torch::Tensor> softmax_sum_forward(torch::Tensor x, torch::Tensor logits);
std::vector<torch::Tensor> softmax_sum_backward(torch::Tensor grad_out, torch::Tensor x, torch::Tensor weights);
"""

def _get_module():
    global _module
    if _module is None:
        _module = load_inline(
            name="softmax_sum_cuda",
            cpp_sources=_cpp_src,
            cuda_sources=_cuda_src,
            functions=["softmax_sum_forward", "softmax_sum_backward"],
            extra_cuda_cflags=["-lineinfo"],
            verbose=True,  # Enable verbose for debugging
        )
    return _module


def softmax_sum(x: torch.Tensor, logits: torch.Tensor):
    """Compute Σ softmax(logits)_i * x_i for x shape [N,B,C,H,W]."""
    if not x.is_cuda or not logits.is_cuda:
        # fallback python
        weights = torch.softmax(logits, 0)
        return (x * weights.view(-1, 1, 1, 1, 1)).sum(dim=0)
    mod: Any = _get_module()
    out, _ = mod.softmax_sum_forward(x.contiguous(), logits.contiguous())
    return out


class SoftmaxSumFunction(torch.autograd.Function):
    """
    Autograd function for fused softmax + weighted sum.
    Forward: out = Σ softmax(logits)_i * x_i
    Backward: efficient computation of gradients w.r.t both x and logits
    """
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, logits: torch.Tensor):
        """
        Args:
            x: [N, B, C, H, W] - N operations, each producing [B,C,H,W] output
            logits: [N] - raw architecture weights (before softmax)
        Returns:
            out: [B, C, H, W] - weighted sum of operations
        """
        if not x.is_cuda or not logits.is_cuda:
            # CPU fallback
            weights = torch.softmax(logits, 0)
            ctx.save_for_backward(x, weights)
            ctx.use_cuda = False
            return (x * weights.view(-1, 1, 1, 1, 1)).sum(dim=0)
        
        # CUDA accelerated path
        mod: Any = _get_module()
        out, weights = mod.softmax_sum_forward(x.contiguous(), logits.contiguous())
        ctx.save_for_backward(x, weights)
        ctx.use_cuda = True
        return out
    
    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        """
        Args:
            grad_out: [B, C, H, W] - gradient w.r.t output
        Returns:
            grad_x: [N, B, C, H, W] - gradient w.r.t input operations
            grad_logits: [N] - gradient w.r.t logits (architecture weights)
        """
        x, weights = ctx.saved_tensors
        
        if not ctx.use_cuda:
            # CPU fallback backward
            grad_x = grad_out.unsqueeze(0) * weights.view(-1, 1, 1, 1, 1)
            grad_w = (x * grad_out.unsqueeze(0)).sum(dim=(1, 2, 3, 4))
            
            # Convert grad_w to grad_logits using softmax gradient
            # d/dlogits softmax(logits) = softmax(logits) * (delta - dot(delta, softmax))
            grad_logits = weights * (grad_w - (grad_w * weights).sum())
            return grad_x, grad_logits
        
        # CUDA accelerated backward
        mod: Any = _get_module()
        grad_x, grad_w = mod.softmax_sum_backward(grad_out.contiguous(), x, weights)
        
        # Convert grad_w to grad_logits
        grad_logits = weights * (grad_w - (grad_w * weights).sum())
        return grad_x, grad_logits


def fused_softmax_sum(x: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """
    High-level interface for fused softmax + weighted sum.
    
    Args:
        x: [N, B, C, H, W] tensor where N is number of operations
        logits: [N] raw architecture weights
    
    Returns:
        [B, C, H, W] weighted combination of operations
    """
    return SoftmaxSumFunction.apply(x, logits)  # type: ignore[return-value] 