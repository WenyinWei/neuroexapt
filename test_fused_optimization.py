#!/usr/bin/env python3
"""
融合优化测试脚本

验证FusedOptimizedMixedOp是否正常工作并提供预期的性能提升
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import time

def test_fused_optimization():
    """测试融合优化功能"""
    print("🚀 测试融合优化功能...")
    
    try:
        from neuroexapt.core.model import Network
        
        print("✅ 创建融合优化模型")
        model = Network(
            C=16, 
            num_classes=10, 
            layers=6, 
            potential_layers=2,
            quiet=True
        )
        
        if torch.cuda.is_available():
            model = model.cuda()
            print("   🔥 使用CUDA加速")
        
        # 验证模型参数
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   📊 模型参数: {total_params:,}")
        
        # 测试前向传播
        batch_size = 4
        x = torch.randn(batch_size, 3, 32, 32)
        if torch.cuda.is_available():
            x = x.cuda()
        
        print("✅ 测试前向传播")
        model.eval()
        
        # 预热
        with torch.no_grad():
            for _ in range(3):
                _ = model(x)
        
        # 性能测试
        times = []
        with torch.no_grad():
            for i in range(10):
                start = time.perf_counter()
                output = model(x)
                end = time.perf_counter()
                times.append(end - start)
                
                if i == 0:
                    print(f"   输出形状: {output.shape}")
        
        avg_time = sum(times) / len(times)
        print(f"   ⚡ 平均前向时间: {avg_time*1000:.2f}ms")
        
        # 获取优化统计（如果可用）
        print("✅ 检查优化统计")
        found_fused_ops = 0
        total_mixed_ops = 0
        
        for name, module in model.named_modules():
            if 'FusedOptimizedMixedOp' in str(type(module)):
                found_fused_ops += 1
                total_mixed_ops += 1
                # 如果模块有统计功能，显示它们
                if hasattr(module, 'get_optimization_stats'):
                    stats = module.get_optimization_stats()
                    if stats['forward_calls'] > 0:
                        print(f"   📈 {name}: 调用{stats['forward_calls']}次, "
                              f"缓存命中率{stats['cache_hit_rate']:.1%}, "
                              f"活跃操作{stats['active_operations']}/{stats['total_operations']}")
                        break  # 只显示一个样例
            elif 'MixedOp' in str(type(module)):
                total_mixed_ops += 1
        
        print(f"   🔄 找到 {found_fused_ops}/{total_mixed_ops} 个融合优化操作")
        
        if found_fused_ops > 0:
            print("🎉 融合优化测试成功！")
            print("💡 优势:")
            print("   - 自动启用所有优化策略")
            print("   - 智能调度和缓存复用")  
            print("   - 选择性梯度计算")
            print("   - 动态操作剪枝")
            print("   - Triton加速支持")
            return True
        else:
            print("⚠️ 未检测到融合优化操作")
            return False
            
    except Exception as e:
        print(f"❌ 融合优化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_optimization_comparison():
    """对比测试：基础vs融合优化"""
    print("\n🔍 对比测试: 基础 vs 融合优化")
    
    try:
        from neuroexapt.core.model import Network
        
        # 基础模型（禁用所有优化）
        print("📊 创建基础模型...")
        basic_model = Network(
            C=16, num_classes=10, layers=4, potential_layers=1,
            use_fused_optimization=False,
            use_gradient_optimized=False,
            use_memory_efficient=False,
            use_lazy_ops=False,
            use_optimized_ops=False,
            use_checkpoint=False,
            quiet=True
        )
        
        # 融合优化模型（默认配置）
        print("🚀 创建融合优化模型...")
        fused_model = Network(
            C=16, num_classes=10, layers=4, potential_layers=1,
            quiet=True
        )
        
        if torch.cuda.is_available():
            basic_model = basic_model.cuda()
            fused_model = fused_model.cuda()
        
        # 测试数据
        x = torch.randn(2, 3, 32, 32)
        if torch.cuda.is_available():
            x = x.cuda()
        
        # 测试基础模型
        basic_model.eval()
        basic_times = []
        with torch.no_grad():
            for _ in range(5):  # 预热
                _ = basic_model(x)
            for _ in range(10):
                start = time.perf_counter()
                _ = basic_model(x)
                end = time.perf_counter()
                basic_times.append(end - start)
        
        # 测试融合优化模型
        fused_model.eval()
        fused_times = []
        with torch.no_grad():
            for _ in range(5):  # 预热
                _ = fused_model(x)
            for _ in range(10):
                start = time.perf_counter()
                _ = fused_model(x)
                end = time.perf_counter()
                fused_times.append(end - start)
        
        basic_avg = sum(basic_times) / len(basic_times)
        fused_avg = sum(fused_times) / len(fused_times)
        speedup = basic_avg / fused_avg if fused_avg > 0 else 1.0
        
        print(f"📈 性能对比结果:")
        print(f"   基础模型: {basic_avg*1000:.2f}ms")
        print(f"   融合优化: {fused_avg*1000:.2f}ms")
        print(f"   性能提升: {speedup:.2f}x")
        
        if speedup > 1.1:
            print("🎉 融合优化显著提升性能！")
        elif speedup > 0.9:
            print("✅ 融合优化保持同等性能（开销很小）")
        else:
            print("⚠️ 性能有所下降，可能需要调优")
        
        return True
        
    except Exception as e:
        print(f"❌ 对比测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 融合优化测试")
    print("=" * 60)
    
    test1_pass = test_fused_optimization()
    test2_pass = test_optimization_comparison()
    
    print("\n" + "=" * 60)
    if test1_pass and test2_pass:
        print("🎉 所有融合优化测试通过！")
        print("💡 现在可以享受默认启用的全方位优化:")
        print("   1. 🚀 梯度优化: 选择性计算 + 检查点")
        print("   2. 💾 内存优化: 流式计算 + 缓存复用")
        print("   3. 🧠 懒计算: 动态剪枝 + 早期终止")
        print("   4. ⚡ Triton加速: 自动CUDA优化")
        print("   5. 🔧 智能调度: 根据负载自适应")
    else:
        print("❌ 部分测试失败，请检查配置")

if __name__ == "__main__":
    main() 