"""
诊断 torch.linalg.matrix_norm 调用错误的问题

这个脚本将测试所有可能导致 linalg.matrix_norm 错误的原因：
1. PyTorch 版本兼容性
2. 输入张量类型和形状
3. 设备兼容性
4. 梯度计算问题
5. 内存问题
"""

import torch
import torch.nn as nn
import numpy as np
import traceback
import sys

def check_pytorch_version():
    """检查PyTorch版本兼容性"""
    print("=" * 60)
    print("1. PyTorch版本检查")
    print("=" * 60)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
        # Skip CUDA version check due to API differences
    
    # 检查 torch.linalg.matrix_norm 是否存在
    try:
        hasattr(torch.linalg, 'matrix_norm')
        print("✅ torch.linalg.matrix_norm 存在")
        
        # 检查版本要求
        version_parts = torch.__version__.split('.')
        major, minor = int(version_parts[0]), int(version_parts[1])
        
        if major < 2:
            print("❌ PyTorch版本过低，需要 >= 2.0.0")
            print("   matrix_norm 在 PyTorch 1.x 中不可用")
            return False
        else:
            print("✅ PyTorch版本符合要求")
            
    except Exception as e:
        print(f"❌ torch.linalg.matrix_norm 检查失败: {e}")
        return False
    
    return True

def test_basic_matrix_norm():
    """测试基本的 matrix_norm 功能"""
    print("\n" + "=" * 60)
    print("2. 基本 matrix_norm 功能测试")
    print("=" * 60)
    
    try:
        # 测试基本功能
        test_matrix = torch.randn(3, 3)
        print(f"测试矩阵: {test_matrix.shape}")
        
        # 测试不同的 norm 类型
        norm_types = ['fro', 'nuc', 2, -2, 1, -1, float('inf'), float('-inf')]
        
        for norm_type in norm_types:
            try:
                result = torch.linalg.matrix_norm(test_matrix, ord=norm_type)
                print(f"✅ norm={norm_type}: {result:.4f}")
            except Exception as e:
                print(f"❌ norm={norm_type} 失败: {e}")
                
    except Exception as e:
        print(f"❌ 基本测试失败: {e}")
        traceback.print_exc()
        return False
    
    return True

def test_tensor_types():
    """测试不同张量类型"""
    print("\n" + "=" * 60)
    print("3. 张量类型测试")
    print("=" * 60)
    
    # 测试不同数据类型
    dtypes = [torch.float32, torch.float64, torch.int32, torch.int64]
    
    for dtype in dtypes:
        try:
            test_tensor = torch.randn(3, 3).to(dtype)
            print(f"测试数据类型: {dtype}")
            
            if dtype in [torch.int32, torch.int64]:
                # 整数类型可能不支持
                try:
                    result = torch.linalg.matrix_norm(test_tensor)
                    print(f"✅ {dtype}: {result:.4f}")
                except Exception as e:
                    print(f"❌ {dtype} 不支持: {e}")
            else:
                result = torch.linalg.matrix_norm(test_tensor)
                print(f"✅ {dtype}: {result:.4f}")
                
        except Exception as e:
            print(f"❌ {dtype} 测试失败: {e}")

def test_tensor_shapes():
    """测试不同张量形状"""
    print("\n" + "=" * 60)
    print("4. 张量形状测试")
    print("=" * 60)
    
    test_shapes = [
        (2, 2),      # 2x2 矩阵
        (3, 3),      # 3x3 矩阵
        (4, 4),      # 4x4 矩阵
        (2, 3),      # 非方阵
        (3, 2),      # 非方阵
        (1, 5),      # 1维类似
        (5, 1),      # 1维类似
        (2, 3, 4),   # 3维张量
        (2, 3, 3),   # 批量矩阵
        (1,),        # 1维张量
        (5,),        # 1维张量
        ()           # 标量
    ]
    
    for shape in test_shapes:
        try:
            test_tensor = torch.randn(*shape) if shape else torch.tensor(5.0)
            print(f"测试形状: {shape}")
            
            result = torch.linalg.matrix_norm(test_tensor)
            print(f"✅ {shape}: {result}")
            
        except Exception as e:
            print(f"❌ {shape} 失败: {e}")

def test_device_compatibility():
    """测试设备兼容性"""
    print("\n" + "=" * 60)
    print("5. 设备兼容性测试")
    print("=" * 60)
    
    # CPU 测试
    try:
        cpu_tensor = torch.randn(3, 3)
        result = torch.linalg.matrix_norm(cpu_tensor)
        print(f"✅ CPU: {result:.4f}")
    except Exception as e:
        print(f"❌ CPU 失败: {e}")
    
    # GPU 测试
    if torch.cuda.is_available():
        try:
            gpu_tensor = torch.randn(3, 3).cuda()
            result = torch.linalg.matrix_norm(gpu_tensor)
            print(f"✅ GPU: {result:.4f}")
        except Exception as e:
            print(f"❌ GPU 失败: {e}")
    else:
        print("❌ GPU 不可用")

def test_gradient_computation():
    """测试梯度计算"""
    print("\n" + "=" * 60)
    print("6. 梯度计算测试")
    print("=" * 60)
    
    try:
        # 测试需要梯度的张量
        test_tensor = torch.randn(3, 3, requires_grad=True)
        print(f"测试张量: {test_tensor.shape}, requires_grad={test_tensor.requires_grad}")
        
        norm_result = torch.linalg.matrix_norm(test_tensor)
        print(f"✅ 计算norm: {norm_result:.4f}")
        
        # 测试反向传播
        loss = norm_result.sum()
        loss.backward()
        print(f"✅ 反向传播成功")
        if test_tensor.grad is not None:
            print(f"梯度形状: {test_tensor.grad.shape}")
        else:
            print("❌ 梯度为None")
        
    except Exception as e:
        print(f"❌ 梯度计算失败: {e}")
        traceback.print_exc()

def test_memory_issues():
    """测试内存相关问题"""
    print("\n" + "=" * 60)
    print("7. 内存问题测试")
    print("=" * 60)
    
    try:
        # 测试大矩阵
        large_sizes = [100, 500, 1000]
        
        for size in large_sizes:
            try:
                large_tensor = torch.randn(size, size)
                result = torch.linalg.matrix_norm(large_tensor)
                print(f"✅ 大小 {size}x{size}: {result:.4f}")
                
                # 清理内存
                del large_tensor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"❌ 大小 {size}x{size} 失败: {e}")
                
    except Exception as e:
        print(f"❌ 内存测试失败: {e}")

def test_special_cases():
    """测试特殊情况"""
    print("\n" + "=" * 60)
    print("8. 特殊情况测试")
    print("=" * 60)
    
    special_cases = [
        ("零矩阵", torch.zeros(3, 3)),
        ("单位矩阵", torch.eye(3)),
        ("包含NaN", torch.tensor([[1., float('nan')], [3., 4.]])),
        ("包含Inf", torch.tensor([[1., float('inf')], [3., 4.]])),
        ("极小值", torch.full((3, 3), 1e-10)),
        ("极大值", torch.full((3, 3), 1e10))
    ]
    
    for name, tensor in special_cases:
        try:
            result = torch.linalg.matrix_norm(tensor)
            print(f"✅ {name}: {result}")
        except Exception as e:
            print(f"❌ {name} 失败: {e}")

def test_existing_code_patterns():
    """测试现有代码中的使用模式"""
    print("\n" + "=" * 60)
    print("9. 现有代码模式测试")
    print("=" * 60)
    
    try:
        # 模拟 radical_architecture_evolution.py 中的使用
        print("测试 NTK 矩阵计算模式...")
        
        # 创建模拟的 NTK 矩阵
        n = 10
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ntk_matrix = torch.randn(n, n, device=device)
        
        # 测试 matrix_rank
        try:
            rank = torch.linalg.matrix_rank(ntk_matrix)
            print(f"✅ matrix_rank: {rank.item()}")
        except Exception as e:
            print(f"❌ matrix_rank 失败: {e}")
            
        # 测试 eigvals
        try:
            eigenvalues = torch.linalg.eigvals(ntk_matrix)
            print(f"✅ eigvals: shape={eigenvalues.shape}")
        except Exception as e:
            print(f"❌ eigvals 失败: {e}")
            
        # 测试 pinv
        try:
            ntk_inv = torch.linalg.pinv(ntk_matrix)
            print(f"✅ pinv: shape={ntk_inv.shape}")
        except Exception as e:
            print(f"❌ pinv 失败: {e}")
            
        # 测试 norm
        try:
            norm_result = torch.norm(ntk_inv)
            print(f"✅ norm: {norm_result.item():.4f}")
        except Exception as e:
            print(f"❌ norm 失败: {e}")
            
    except Exception as e:
        print(f"❌ 现有代码模式测试失败: {e}")
        traceback.print_exc()

def main():
    """主函数"""
    print("🔍 torch.linalg.matrix_norm 错误诊断")
    print("这个脚本将检查所有可能导致错误的原因")
    print()
    
    try:
        # 1. 版本检查
        if not check_pytorch_version():
            print("\n❌ 版本检查失败，请升级PyTorch到2.0+")
            return
            
        # 2. 基本功能测试
        if not test_basic_matrix_norm():
            print("\n❌ 基本功能测试失败")
            return
            
        # 3. 其他测试
        test_tensor_types()
        test_tensor_shapes()
        test_device_compatibility()
        test_gradient_computation()
        test_memory_issues()
        test_special_cases()
        test_existing_code_patterns()
        
        print("\n" + "=" * 60)
        print("✅ 诊断完成")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 诊断过程中发生错误: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 