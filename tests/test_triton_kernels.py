import pytest
import torch

from neuroexapt.kernels import TRITON_AVAILABLE, sepconv_forward_generic


@pytest.mark.parametrize("kernel_size,stride,dilation", [
    (3, 1, 1),
    (3, 2, 1),
    (3, 1, 2),
])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_sepconv_correctness(kernel_size, stride, dilation):
    B, C, H, W = 2, 16, 32, 32
    x = torch.randn(B, C, H, W, device="cuda")

    dw_weight = torch.randn(C, 1, kernel_size, kernel_size, device="cuda")
    pw_weight = torch.randn(C * 2, C, 1, 1, device="cuda")
    bias = torch.randn(C * 2, device="cuda")

    y_ref = torch.nn.functional.conv2d(
        torch.nn.functional.conv2d(
            x,
            dw_weight,
            None,
            stride=stride,
            padding=((kernel_size - 1) * dilation) // 2,
            dilation=dilation,
            groups=C,
        ),
        pw_weight,
        bias,
        stride=1,
        padding=0,
    )

    y_t = sepconv_forward_generic(
        x,
        dw_weight,
        pw_weight,
        bias,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
    )

    tol = 1e-4 if x.dtype == torch.float32 else 5e-3
    max_err = (y_ref - y_t).abs().max().item()
    assert max_err < tol, f"Mismatch: {max_err}"


@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not installed")
def test_triton_available():
    # Simple sanity check that Triton was imported when expected.
    assert TRITON_AVAILABLE 