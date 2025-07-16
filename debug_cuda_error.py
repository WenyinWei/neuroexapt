#!/usr/bin/env python3
"""
CUDA错误深度调试脚本
找出真正的问题所在，而不是简单禁用功能
"""

import torch
import sys
import os
import traceback
import warnings

# Add the project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_cuda_environment():
    """检查CUDA环境状态"""
    print("🔍 CUDA环境检查:")
    print(f"   PyTorch版本: {torch.__version__}")
    print(f"   CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   CUDA版本: {torch.version.cuda}")
        print(f"   GPU数量: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mem_total = props.total_memory / 1024**3
            mem_alloc = torch.cuda.memory_allocated(i) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
            
            print(f"   GPU {i}: {props.name}")
            print(f"     总内存: {mem_total:.2f}GB")
            print(f"     已分配: {mem_alloc:.2f}GB")
            print(f"     已保留: {mem_reserved:.2f}GB")
            print(f"     计算能力: {props.major}.{props.minor}")

def test_basic_operations():
    """测试基本CUDA操作"""
    print("\n🧪 基本CUDA操作测试:")
    
    try:
        # 1. 基本张量操作
        x = torch.randn(4, 16, 32, 32, device='cuda')
        y = torch.randn(4, 16, 32, 32, device='cuda')
        z = x + y
        print("✅ 基本张量运算正常")
        
        # 2. 基本卷积操作
        conv = torch.nn.Conv2d(16, 32, 3, padding=1).cuda()
        out = conv(x)
        print("✅ 基本卷积操作正常")
        
        # 3. 分组卷积（depthwise）
        dw_conv = torch.nn.Conv2d(16, 16, 3, padding=1, groups=16).cuda()
        dw_out = dw_conv(x)
        print("✅ 分组卷积操作正常")
        
        # 4. 内存清理
        del x, y, z, out, dw_out
        torch.cuda.empty_cache()
        print("✅ 内存清理正常")
        
        return True
        
    except Exception as e:
        print(f"❌ 基本操作失败: {e}")
        traceback.print_exc()
        return False

def test_sepconv_step_by_step():
    """逐步测试分离卷积，找出确切的问题点"""
    print("\n🔬 分离卷积逐步调试:")
    
    try:
        # 创建测试数据 - 使用较小的尺寸
        B, C, H, W = 2, 8, 16, 16
        print(f"   测试尺寸: B={B}, C={C}, H={H}, W={W}")
        
        # Step 1: 创建输入张量
        x = torch.randn(B, C, H, W, device='cuda', dtype=torch.float32)
        print(f"✅ 输入张量创建成功: {x.shape}, device={x.device}, dtype={x.dtype}")
        print(f"   内存检查: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
        
        # Step 2: 创建深度卷积权重
        dw_weight = torch.randn(C, 1, 3, 3, device='cuda', dtype=torch.float32)
        print(f"✅ 深度卷积权重创建成功: {dw_weight.shape}")
        
        # Step 3: 创建点卷积权重
        C_out = 16
        pw_weight = torch.randn(C_out, C, 1, 1, device='cuda', dtype=torch.float32)
        print(f"✅ 点卷积权重创建成功: {pw_weight.shape}")
        
        # Step 4: 执行深度卷积
        print("\n📍 执行深度卷积...")
        dw_out = torch.nn.functional.conv2d(
            x, dw_weight, bias=None, stride=1, 
            padding=1, groups=C
        )
        print(f"✅ 深度卷积成功: {dw_out.shape}")
        print(f"   内存检查: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
        
        # Step 5: 执行点卷积
        print("\n📍 执行点卷积...")
        pw_out = torch.nn.functional.conv2d(dw_out, pw_weight, bias=None)
        print(f"✅ 点卷积成功: {pw_out.shape}")
        print(f"   内存检查: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
        
        # Step 6: 测试带bias的点卷积
        print("\n📍 测试带bias的点卷积...")
        bias = torch.randn(C_out, device='cuda', dtype=torch.float32)
        pw_out_bias = torch.nn.functional.conv2d(dw_out, pw_weight, bias=bias)
        print(f"✅ 带bias的点卷积成功: {pw_out_bias.shape}")
        
        # Step 7: 清理内存
        del x, dw_weight, pw_weight, dw_out, pw_out, bias, pw_out_bias
        torch.cuda.empty_cache()
        print("✅ 内存清理完成")
        
        return True
        
    except Exception as e:
        print(f"❌ 分离卷积失败: {e}")
        print(f"❌ 错误类型: {type(e).__name__}")
        traceback.print_exc()
        
        # 打印详细的tensor信息
        import gc
        for obj in gc.get_objects():
            if torch.is_tensor(obj) and obj.is_cuda:
                print(f"   未清理的tensor: {obj.shape}, {obj.dtype}, {obj.device}")
        
        return False

def test_triton_sepconv():
    """测试Triton分离卷积的具体实现"""
    print("\n⚡ Triton分离卷积调试:")
    
    try:
        from neuroexapt.kernels.sepconv_triton import sepconv_forward_generic, TRITON_AVAILABLE, _TRITON_DISABLED
        
        print(f"   Triton可用: {TRITON_AVAILABLE}")
        print(f"   Triton禁用: {_TRITON_DISABLED}")
        
        if not TRITON_AVAILABLE:
            print("⚠️ Triton不可用，将测试PyTorch fallback路径")
        
        # 创建测试数据
        B, C, H, W = 2, 8, 16, 16
        x = torch.randn(B, C, H, W, device='cuda', dtype=torch.float32)
        dw_weight = torch.randn(C, 1, 3, 3, device='cuda', dtype=torch.float32)
        pw_weight = torch.randn(16, C, 1, 1, device='cuda', dtype=torch.float32)
        
        print(f"   输入形状: x={x.shape}, dw={dw_weight.shape}, pw={pw_weight.shape}")
        
        # 确保张量是连续的
        if not x.is_contiguous():
            x = x.contiguous()
            print("   输入张量已转为连续")
        
        if not dw_weight.is_contiguous():
            dw_weight = dw_weight.contiguous()
            print("   深度权重已转为连续")
            
        if not pw_weight.is_contiguous():
            pw_weight = pw_weight.contiguous()
            print("   点权重已转为连续")
        
        # 执行sepconv
        print("\n📍 执行Triton/PyTorch sepconv...")
        result = sepconv_forward_generic(x, dw_weight, pw_weight, bias=None)
        print(f"✅ Sepconv成功: {result.shape}")
        
        # 测试带bias
        bias = torch.randn(16, device='cuda', dtype=torch.float32)
        result_bias = sepconv_forward_generic(x, dw_weight, pw_weight, bias=bias)
        print(f"✅ 带bias的Sepconv成功: {result_bias.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Triton sepconv失败: {e}")
        print(f"❌ 错误类型: {type(e).__name__}")
        traceback.print_exc()
        return False

def test_operations_integration():
    """测试operations.py中的SepConv集成"""
    print("\n🧩 SepConv模块集成测试:")
    
    try:
        from neuroexapt.core.operations import SepConv
        
        # 创建SepConv模块
        sepconv_module = SepConv(8, 16, 3, 1, 1, affine=True).cuda()
        print("✅ SepConv模块创建成功")
        
        # 创建输入
        x = torch.randn(2, 8, 16, 16, device='cuda')
        print(f"   输入形状: {x.shape}")
        
        # 执行前向传播
        print("\n📍 执行SepConv前向传播...")
        output = sepconv_module(x)
        print(f"✅ SepConv前向传播成功: {output.shape}")
        
        # 测试梯度
        print("\n📍 测试梯度计算...")
        loss = output.sum()
        loss.backward()
        print("✅ 梯度计算成功")
        
        return True
        
    except Exception as e:
        print(f"❌ SepConv模块测试失败: {e}")
        traceback.print_exc()
        return False

def test_with_exact_error_conditions():
    """使用导致错误的确切条件进行测试"""
    print("\n🎯 原始错误条件重现测试:")
    
    try:
        # 使用原始的参数
        from neuroexapt.core.model import Network
        
        print("   创建原始大小的网络...")
        model = Network(C=32, num_classes=10, layers=16, potential_layers=4).cuda()
        print(f"✅ 网络创建成功: {sum(p.numel() for p in model.parameters())} 参数")
        
        # 创建原始大小的输入
        batch_size = 64
        x = torch.randn(batch_size, 3, 32, 32, device='cuda')
        print(f"   输入形状: {x.shape}")
        
        # 执行前向传播
        print("\n📍 执行网络前向传播...")
        
        # 监控内存使用
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated()
        
        output = model(x)
        
        peak_memory = torch.cuda.max_memory_allocated()
        final_memory = torch.cuda.memory_allocated()
        
        print(f"✅ 网络前向传播成功: {output.shape}")
        print(f"   内存使用: 初始{initial_memory/1024**2:.1f}MB -> 峰值{peak_memory/1024**2:.1f}MB -> 最终{final_memory/1024**2:.1f}MB")
        
        return True
        
    except Exception as e:
        print(f"❌ 原始条件测试失败: {e}")
        traceback.print_exc()
        
        # 检查可能的原因
        print("\n🔍 可能的失败原因分析:")
        print(f"   当前GPU内存使用: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
        print(f"   当前GPU内存缓存: {torch.cuda.memory_reserved()/1024**2:.1f}MB")
        
        props = torch.cuda.get_device_properties(0)
        total_memory = props.total_memory / 1024**2
        print(f"   GPU总内存: {total_memory:.1f}MB")
        
        if torch.cuda.memory_allocated() / props.total_memory > 0.8:
            print("⚠️ 可能是内存不足导致的问题")
        
        return False

def main():
    """主调试函数"""
    print("🔍 CUDA错误深度调试")
    print("=" * 60)
    print("目标: 找出真正的问题所在，保持Triton优化")
    print("=" * 60)
    
    # 设置警告过滤
    warnings.filterwarnings('error', category=UserWarning)
    
    test_results = {}
    
    # 1. 环境检查
    check_cuda_environment()
    
    # 2. 基本操作测试
    test_results['basic_ops'] = test_basic_operations()
    
    # 3. 分离卷积逐步测试
    test_results['sepconv_steps'] = test_sepconv_step_by_step()
    
    # 4. Triton分离卷积测试
    test_results['triton_sepconv'] = test_triton_sepconv()
    
    # 5. SepConv模块集成测试
    test_results['sepconv_module'] = test_operations_integration()
    
    # 6. 原始错误条件测试
    test_results['original_conditions'] = test_with_exact_error_conditions()
    
    # 结果分析
    print("\n" + "=" * 60)
    print("📊 调试结果总结:")
    print("=" * 60)
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {test_name}: {status}")
    
    print(f"\n通过率: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
    
    # 根据结果给出建议
    if all(test_results.values()):
        print("\n🎉 所有测试通过！原始CUDA错误可能已经解决。")
        print("💡 建议: 尝试运行原始的分离训练命令")
        print("   python examples/basic_classification.py --mode separated --epochs 10 --batch_size 32")
    
    elif test_results['basic_ops'] and test_results['sepconv_steps']:
        print("\n⚠️ 基本操作正常，但高级功能有问题")
        print("💡 建议: 检查Triton内核实现或模块集成问题")
    
    elif not test_results['basic_ops']:
        print("\n❌ 基本CUDA操作失败")
        print("💡 建议: 检查CUDA驱动、PyTorch安装或GPU硬件问题")
    
    else:
        print("\n🔍 部分测试失败，需要进一步调查")
        
        # 识别失败模式
        if not test_results['original_conditions']:
            print("💡 可能是大型网络的内存或计算问题")
        if not test_results['triton_sepconv']:
            print("💡 可能是Triton内核的具体实现问题")
        if not test_results['sepconv_module']:
            print("💡 可能是PyTorch模块集成问题")

if __name__ == "__main__":
    main() 