#!/usr/bin/env python3
"""
测试优化模块的导入和基本功能
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """测试所有优化模块的导入"""
    print("🔧 Testing optimized module imports...")
    
    try:
        # 测试core模块导入
        print("📦 Testing core.fast_operations...")
        from neuroexapt.core.fast_operations import (
            FastMixedOp, BatchedArchitectureUpdate, MemoryEfficientCell,
            FastDeviceManager, get_fast_device_manager, OperationProfiler
        )
        print("✅ Core fast_operations imported successfully")
        
        # 测试math模块导入
        print("📦 Testing math.fast_math...")
        from neuroexapt.math.fast_math import (
            FastEntropy, FastGradients, FastNumerical, FastStatistics,
            PerformanceProfiler, profile_op
        )
        print("✅ Math fast_math imported successfully")
        
        # 测试基础组件
        print("📦 Testing basic components...")
        from neuroexapt.core.genotypes import PRIMITIVES
        print("✅ PRIMITIVES imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_basic_functionality():
    """测试基本功能"""
    print("\n🧪 Testing basic functionality...")
    
    try:
        import torch
        from neuroexapt.core.fast_operations import FastMixedOp
        from neuroexapt.math.fast_math import FastEntropy
        
        # 测试FastMixedOp
        print("🔧 Testing FastMixedOp...")
        mixed_op = FastMixedOp(32, stride=1)
        test_input = torch.randn(4, 32, 16, 16)
        
        # 创建测试权重
        from neuroexapt.core.genotypes import PRIMITIVES
        num_ops = len(PRIMITIVES)
        test_weights = torch.softmax(torch.randn(num_ops), dim=0)
        
        # 前向传播测试
        output = mixed_op(test_input, test_weights, training=True)
        print(f"   Input shape: {test_input.shape}")
        print(f"   Output shape: {output.shape}")
        print("✅ FastMixedOp test passed")
        
        # 测试FastEntropy
        print("🔧 Testing FastEntropy...")
        test_logits = torch.randn(4, 10)
        entropy = FastEntropy.entropy_jit(test_logits)
        print(f"   Logits shape: {test_logits.shape}")
        print(f"   Entropy shape: {entropy.shape}")
        print("✅ FastEntropy test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 ASO-SE Optimized Modules Test")
    print("=" * 50)
    
    # 测试导入
    import_success = test_imports()
    
    if import_success:
        # 测试功能
        func_success = test_basic_functionality()
        
        if func_success:
            print("\n🎉 All tests passed! Ready to run optimized training.")
            print("\n💡 Try running:")
            print("   python examples/aso_se_classification_optimized.py --cycles 5 --batch_size 64")
        else:
            print("\n❌ Functionality tests failed. Please check implementation.")
    else:
        print("\n❌ Import tests failed. Please check dependencies.")

if __name__ == "__main__":
    main()