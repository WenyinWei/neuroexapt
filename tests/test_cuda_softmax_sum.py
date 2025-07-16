import pytest
import torch
import numpy as np

from neuroexapt.cuda_ops import CUDA_AVAILABLE
if CUDA_AVAILABLE:
    from neuroexapt.cuda_ops import SoftmaxSumFunction


class TestSoftmaxSum:
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_softmax_sum_correctness(self):
        """Test numerical correctness of SoftmaxSum against PyTorch reference."""
        N, B, C, H, W = 8, 2, 16, 32, 32
        torch.manual_seed(42)
        
        x = torch.randn(N, B, C, H, W, device="cuda", requires_grad=True)
        logits = torch.randn(N, device="cuda", requires_grad=True)
        
        # Reference computation
        weights_ref = torch.softmax(logits, 0)
        out_ref = (x * weights_ref.view(-1, 1, 1, 1, 1)).sum(dim=0)
        
        # CUDA implementation
        out_cuda: torch.Tensor = SoftmaxSumFunction.apply(x.clone().detach().requires_grad_(True), 
                                          logits.clone().detach().requires_grad_(True))  # type: ignore[assignment]
        
        # Forward pass comparison
        assert torch.allclose(out_ref, out_cuda, atol=1e-4, rtol=1e-4), \
            f"Forward mismatch: max_diff={torch.max(torch.abs(out_ref - out_cuda))}"  # type: ignore[operator]
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available") 
    def test_softmax_sum_backward(self):
        """Test backward pass gradients."""
        N, B, C, H, W = 6, 2, 8, 16, 16
        torch.manual_seed(123)
        
        x = torch.randn(N, B, C, H, W, device="cuda", requires_grad=True)
        logits = torch.randn(N, device="cuda", requires_grad=True)
        
        # Reference gradients
        weights_ref = torch.softmax(logits, 0)
        out_ref = (x * weights_ref.view(-1, 1, 1, 1, 1)).sum(dim=0)
        loss_ref = out_ref.sum()
        loss_ref.backward()
        assert x.grad is not None and logits.grad is not None
        grad_x_ref = x.grad.clone()
        grad_logits_ref = logits.grad.clone()
        
        # Reset gradients
        x.grad = None
        logits.grad = None
        
        # CUDA gradients 
        out_cuda: torch.Tensor = SoftmaxSumFunction.apply(x, logits)  # type: ignore[assignment]
        loss_cuda = out_cuda.sum()  # type: ignore[union-attr]
        loss_cuda.backward()
        assert x.grad is not None and logits.grad is not None
        grad_x_cuda = x.grad  # type: ignore[assignment]
        grad_logits_cuda = logits.grad  # type: ignore[assignment]
        
        # Compare gradients
        assert torch.allclose(grad_x_ref, grad_x_cuda, atol=1e-4, rtol=1e-4), \
            f"grad_x mismatch: max_diff={torch.max(torch.abs(grad_x_ref - grad_x_cuda))}"  # type: ignore[operator]
        assert torch.allclose(grad_logits_ref, grad_logits_cuda, atol=1e-4, rtol=1e-4), \
            f"grad_logits mismatch: max_diff={torch.max(torch.abs(grad_logits_ref - grad_logits_cuda))}"  # type: ignore[operator]
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_softmax_sum_cpu_fallback(self):
        """Test CPU fallback when tensors are on CPU."""
        N, B, C, H, W = 4, 1, 8, 8, 8
        
        x = torch.randn(N, B, C, H, W, requires_grad=True)  # CPU tensor
        logits = torch.randn(N, requires_grad=True)
        
        # Should fallback to PyTorch implementation
        out: torch.Tensor = SoftmaxSumFunction.apply(x, logits)  # type: ignore[assignment]
        
        # Reference
        weights = torch.softmax(logits, 0)
        out_ref = (x * weights.view(-1, 1, 1, 1, 1)).sum(dim=0)
        
        assert torch.allclose(out, out_ref, atol=1e-6)
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_softmax_sum_shapes(self):
        """Test various input shapes."""
        test_shapes = [
            (3, 1, 4, 8, 8),
            (5, 4, 16, 16, 16),
            (8, 2, 32, 32, 32)
        ]
        
        for N, B, C, H, W in test_shapes:
            x = torch.randn(N, B, C, H, W, device="cuda")
            logits = torch.randn(N, device="cuda")
            
            out: torch.Tensor = SoftmaxSumFunction.apply(x, logits)  # type: ignore[assignment]
            assert out.shape == (B, C, H, W), f"Wrong output shape for input {(N,B,C,H,W)}"
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_extreme_weights(self):
        """Test with extreme weight distributions."""
        N, B, C, H, W = 6, 2, 8, 16, 16
        
        x = torch.randn(N, B, C, H, W, device="cuda")
        
        # Test highly concentrated weights
        logits_concentrated = torch.tensor([-10., 10., -10., -10., -10., -10.], device="cuda")
        out: torch.Tensor = SoftmaxSumFunction.apply(x, logits_concentrated)  # type: ignore[assignment]
        
        # Should be close to x[1] since weight[1] ≈ 1
        expected = x[1]
        assert torch.allclose(out, expected, atol=1e-3)
        
        # Test uniform weights
        logits_uniform = torch.zeros(N, device="cuda")
        out_uniform: torch.Tensor = SoftmaxSumFunction.apply(x, logits_uniform)  # type: ignore[assignment]
        
        # Should be close to mean
        expected_uniform = x.mean(dim=0)
        assert torch.allclose(out_uniform, expected_uniform, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 