"""
简化的 torch.linalg.matrix_norm 错误诊断脚本
"""

import torch
import torch.nn as nn
import traceback
import sys

def check_pytorch_version():
    """检查PyTorch版本兼容性"""
    print("=" * 60)
    print("PyTorch版本检查")
    print("=" * 60)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # 检查版本要求
    version_parts = torch.__version__.split('.')
    major = int(version_parts[0])
    
    if major < 2:
        print("ERROR: PyTorch version is too old, need >= 2.0.0")
        return False
    else:
        print("OK: PyTorch version is compatible")
        
    return True

def test_matrix_norm_basic():
    """测试基本的 matrix_norm 功能"""
    print("\n" + "=" * 60)
    print("基本 matrix_norm 功能测试")
    print("=" * 60)
    
    try:
        # 测试基本功能
        test_matrix = torch.randn(3, 3)
        print(f"Test matrix shape: {test_matrix.shape}")
        print(f"Test matrix dtype: {test_matrix.dtype}")
        
        # 尝试调用 matrix_norm
        result = torch.linalg.matrix_norm(test_matrix)
        print(f"OK: matrix_norm result: {result:.4f}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Basic matrix_norm test failed: {e}")
        traceback.print_exc()
        return False

def test_common_failure_cases():
    """测试常见的失败案例"""
    print("\n" + "=" * 60)
    print("常见失败案例测试")
    print("=" * 60)
    
    # 测试整数张量
    try:
        int_tensor = torch.randint(0, 10, (3, 3))
        result = torch.linalg.matrix_norm(int_tensor)
        print(f"ERROR: Integer tensor should fail but didn't: {result}")
    except Exception as e:
        print(f"OK: Integer tensor correctly failed: {e}")
    
    # 测试1维张量
    try:
        vec_tensor = torch.randn(5)
        result = torch.linalg.matrix_norm(vec_tensor)
        print(f"ERROR: 1D tensor should fail but didn't: {result}")
    except Exception as e:
        print(f"OK: 1D tensor correctly failed: {e}")
    
    # 测试0维张量
    try:
        scalar_tensor = torch.tensor(5.0)
        result = torch.linalg.matrix_norm(scalar_tensor)
        print(f"ERROR: Scalar tensor should fail but didn't: {result}")
    except Exception as e:
        print(f"OK: Scalar tensor correctly failed: {e}")

def test_model_weight_norms():
    """测试模型权重的norm计算"""
    print("\n" + "=" * 60)
    print("模型权重norm测试")
    print("=" * 60)
    
    # 创建简单模型
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(128, 10)
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # 测试各层权重的norm
    for name, param in model.named_parameters():
        try:
            if param.dim() >= 2:
                # 对于2维以上的参数，尝试计算matrix_norm
                if param.dim() == 2:
                    norm_result = torch.linalg.matrix_norm(param)
                    print(f"OK: {name} shape={param.shape}, norm={norm_result:.4f}")
                elif param.dim() == 4:  # Conv2d weights
                    # 对于4维卷积权重，需要reshape
                    weight_2d = param.view(param.size(0), -1)
                    norm_result = torch.linalg.matrix_norm(weight_2d)
                    print(f"OK: {name} shape={param.shape}, reshaped={weight_2d.shape}, norm={norm_result:.4f}")
                else:
                    print(f"SKIP: {name} shape={param.shape} (>2D, complex case)")
            else:
                print(f"SKIP: {name} shape={param.shape} (1D bias)")
                
        except Exception as e:
            print(f"ERROR: {name} failed: {e}")

def test_gradient_computation():
    """测试梯度计算"""
    print("\n" + "=" * 60)
    print("梯度计算测试")
    print("=" * 60)
    
    try:
        # 创建需要梯度的张量
        test_tensor = torch.randn(3, 3, requires_grad=True)
        print(f"Test tensor: shape={test_tensor.shape}, requires_grad={test_tensor.requires_grad}")
        
        # 计算norm
        norm_result = torch.linalg.matrix_norm(test_tensor)
        print(f"OK: norm computed: {norm_result:.4f}")
        
        # 反向传播
        loss = norm_result * 2
        loss.backward()
        
        if test_tensor.grad is not None:
            print(f"OK: gradient computed, shape={test_tensor.grad.shape}")
        else:
            print("ERROR: gradient is None")
            
    except Exception as e:
        print(f"ERROR: gradient computation failed: {e}")
        traceback.print_exc()

def test_device_issues():
    """测试设备相关问题"""
    print("\n" + "=" * 60)
    print("设备兼容性测试")
    print("=" * 60)
    
    # CPU测试
    try:
        cpu_tensor = torch.randn(3, 3)
        cpu_result = torch.linalg.matrix_norm(cpu_tensor)
        print(f"OK: CPU tensor: {cpu_result:.4f}")
    except Exception as e:
        print(f"ERROR: CPU tensor failed: {e}")
    
    # GPU测试
    if torch.cuda.is_available():
        try:
            gpu_tensor = torch.randn(3, 3).cuda()
            gpu_result = torch.linalg.matrix_norm(gpu_tensor)
            print(f"OK: GPU tensor: {gpu_result:.4f}")
        except Exception as e:
            print(f"ERROR: GPU tensor failed: {e}")
    else:
        print("SKIP: GPU not available")

def test_ntk_simulation():
    """测试NTK矩阵计算模拟"""
    print("\n" + "=" * 60)
    print("NTK矩阵计算模拟")
    print("=" * 60)
    
    try:
        # 模拟 radical_architecture_evolution.py 中的代码
        n = 10
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建模拟的NTK矩阵
        ntk_matrix = torch.randn(n, n, device=device)
        print(f"NTK matrix: shape={ntk_matrix.shape}, device={ntk_matrix.device}")
        
        # 测试matrix_rank
        try:
            rank = torch.linalg.matrix_rank(ntk_matrix)
            print(f"OK: matrix_rank: {rank.item()}")
        except Exception as e:
            print(f"ERROR: matrix_rank failed: {e}")
            
        # 测试eigvals
        try:
            eigenvalues = torch.linalg.eigvals(ntk_matrix)
            print(f"OK: eigvals: shape={eigenvalues.shape}")
        except Exception as e:
            print(f"ERROR: eigvals failed: {e}")
            
        # 测试pinv
        try:
            ntk_inv = torch.linalg.pinv(ntk_matrix)
            print(f"OK: pinv: shape={ntk_inv.shape}")
            
            # 测试普通norm
            norm_result = torch.norm(ntk_inv)
            print(f"OK: norm of pinv: {norm_result.item():.4f}")
            
        except Exception as e:
            print(f"ERROR: pinv failed: {e}")
            
    except Exception as e:
        print(f"ERROR: NTK simulation failed: {e}")
        traceback.print_exc()

def main():
    """主函数"""
    print("torch.linalg.matrix_norm 错误诊断")
    print("=" * 60)
    
    try:
        # 版本检查
        if not check_pytorch_version():
            print("\nERROR: Version check failed")
            return False
            
        # 基本功能测试
        if not test_matrix_norm_basic():
            print("\nERROR: Basic functionality test failed")
            return False
            
        # 其他测试
        test_common_failure_cases()
        test_model_weight_norms()
        test_gradient_computation()
        test_device_issues()
        test_ntk_simulation()
        
        print("\n" + "=" * 60)
        print("诊断完成")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\nERROR: Diagnosis failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 