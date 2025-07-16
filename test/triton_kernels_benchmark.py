#!/usr/bin/env python3
"""
NeuroExapt 现有Triton内核基准测试

专门测试项目中已实现的Triton优化内核的性能：
1. CUDA SoftmaxSum 扩展
2. Triton 分离卷积内核
3. Triton 池化内核
4. MixedOp 优化
"""

import torch
import torch.nn.functional as F
import time
import json
import os
import sys
from typing import Dict, List, Tuple
from datetime import datetime
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def benchmark_cuda_softmax_sum():
    """基准测试CUDA SoftmaxSum扩展"""
    print("🔥 测试 CUDA SoftmaxSum 扩展")
    print("-" * 40)
    
    try:
        from neuroexapt.cuda_ops import SoftmaxSumFunction, CUDA_AVAILABLE
        print(f"CUDA可用: {CUDA_AVAILABLE}")
    except ImportError as e:
        print(f"❌ 无法导入CUDA模块: {e}")
        return {}
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"测试设备: {device}")
    
    # 测试配置 - 模拟典型的NAS场景
    test_configs = [
        {"name": "Small", "N": 4, "B": 2, "C": 16, "H": 16, "W": 16},
        {"name": "Medium", "N": 8, "B": 4, "C": 32, "H": 32, "W": 32},
        {"name": "Large", "N": 12, "B": 8, "C": 64, "H": 64, "W": 64},
        {"name": "XLarge", "N": 16, "B": 16, "C": 128, "H": 128, "W": 128},
    ]
    
    results = {}
    warmup_runs = 5
    benchmark_runs = 20
    
    def pytorch_softmax_sum(x, logits):
        """PyTorch参考实现"""
        weights = torch.softmax(logits, 0)
        return (x * weights.view(-1, 1, 1, 1, 1)).sum(dim=0)
    
    for config in test_configs:
        print(f"\n🧪 测试配置: {config['name']}")
        N, B, C, H, W = config["N"], config["B"], config["C"], config["H"], config["W"]
        
        # 生成测试数据
        x = torch.randn(N, B, C, H, W, device=device, requires_grad=True)
        logits = torch.randn(N, device=device, requires_grad=True)
        
        print(f"输入形状: x={list(x.shape)}, logits={list(logits.shape)}")
        print(f"总元素数: {x.numel():,}")
        print(f"内存使用: {x.numel() * 4 / 1024 / 1024:.1f} MB")
        
        try:
            # 预热
            for _ in range(warmup_runs):
                _ = pytorch_softmax_sum(x, logits)
                _ = SoftmaxSumFunction.apply(x, logits)
            
            # 基准测试 PyTorch
            if device == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            for _ in range(benchmark_runs):
                pytorch_result = pytorch_softmax_sum(x, logits)
            if device == "cuda":
                torch.cuda.synchronize()
            pytorch_time = (time.perf_counter() - start_time) / benchmark_runs * 1000
            
            # 基准测试 CUDA优化版本
            if device == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            for _ in range(benchmark_runs):
                cuda_result = SoftmaxSumFunction.apply(x, logits)
            if device == "cuda":
                torch.cuda.synchronize()
            cuda_time = (time.perf_counter() - start_time) / benchmark_runs * 1000
            
            # 验证正确性
            max_diff = torch.max(torch.abs(pytorch_result - cuda_result)).item()
            is_correct = max_diff < 1e-3
            
            # 计算加速比
            speedup = pytorch_time / cuda_time if cuda_time > 0 else 0
            
            # 内存使用分析
            pytorch_memory = x.numel() * x.element_size() * 2  # x + intermediate results
            cuda_memory = x.numel() * x.element_size()  # 融合操作减少中间结果
            memory_savings = (pytorch_memory - cuda_memory) / pytorch_memory * 100
            
            result = {
                'config': config,
                'pytorch_time_ms': pytorch_time,
                'cuda_time_ms': cuda_time,
                'speedup': speedup,
                'max_difference': max_diff,
                'is_correct': is_correct,
                'memory_savings_percent': memory_savings,
                'device': device
            }
            
            results[config['name']] = result
            
            print(f"✅ PyTorch: {pytorch_time:.2f}ms")
            print(f"⚡ CUDA: {cuda_time:.2f}ms")
            print(f"🚀 加速比: {speedup:.2f}x")
            print(f"🎯 数值误差: {max_diff:.2e} {'✅' if is_correct else '❌'}")
            print(f"💾 内存节省: {memory_savings:.1f}%")
            
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            results[config['name']] = {'error': str(e)}
    
    return results

def benchmark_triton_sepconv():
    """基准测试Triton分离卷积内核"""
    print("\n⚡ 测试 Triton 分离卷积内核")
    print("-" * 40)
    
    try:
        from neuroexapt.kernels import sepconv_forward_generic, TRITON_AVAILABLE
        print(f"Triton可用: {TRITON_AVAILABLE}")
    except ImportError as e:
        print(f"❌ 无法导入Triton模块: {e}")
        return {}
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 测试配置
    test_configs = [
        {"name": "Small", "B": 2, "C": 16, "H": 32, "W": 32, "K": 3},
        {"name": "Medium", "B": 4, "C": 32, "H": 64, "W": 64, "K": 3},
        {"name": "Large", "B": 8, "C": 64, "H": 128, "W": 128, "K": 3},
    ]
    
    results = {}
    warmup_runs = 5
    benchmark_runs = 10
    
    def pytorch_sepconv(x, dw_weight, pw_weight, bias=None):
        """PyTorch参考实现"""
        # 深度卷积
        y = F.conv2d(x, dw_weight, bias=None, stride=1, padding=1, groups=x.size(1))
        # 点卷积
        y = F.conv2d(y, pw_weight, bias=bias)
        return y
    
    for config in test_configs:
        print(f"\n🧪 测试配置: {config['name']}")
        B, C, H, W, K = config["B"], config["C"], config["H"], config["W"], config["K"]
        
        # 生成测试数据
        x = torch.randn(B, C, H, W, device=device, requires_grad=True)
        dw_weight = torch.randn(C, 1, K, K, device=device, requires_grad=True)
        pw_weight = torch.randn(C*2, C, 1, 1, device=device, requires_grad=True)
        bias = torch.randn(C*2, device=device, requires_grad=True)
        
        print(f"输入形状: x={list(x.shape)}")
        print(f"DW权重: {list(dw_weight.shape)}")
        print(f"PW权重: {list(pw_weight.shape)}")
        
        try:
            # 预热
            for _ in range(warmup_runs):
                _ = pytorch_sepconv(x, dw_weight, pw_weight, bias)
                _ = sepconv_forward_generic(x, dw_weight, pw_weight, bias)
            
            # 基准测试 PyTorch
            if device == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            for _ in range(benchmark_runs):
                pytorch_result = pytorch_sepconv(x, dw_weight, pw_weight, bias)
            if device == "cuda":
                torch.cuda.synchronize()
            pytorch_time = (time.perf_counter() - start_time) / benchmark_runs * 1000
            
            # 基准测试 Triton
            if device == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            for _ in range(benchmark_runs):
                triton_result = sepconv_forward_generic(x, dw_weight, pw_weight, bias)
            if device == "cuda":
                torch.cuda.synchronize()
            triton_time = (time.perf_counter() - start_time) / benchmark_runs * 1000
            
            # 验证正确性
            max_diff = torch.max(torch.abs(pytorch_result - triton_result)).item()
            is_correct = max_diff < 1e-3
            
            speedup = pytorch_time / triton_time if triton_time > 0 else 0
            
            result = {
                'config': config,
                'pytorch_time_ms': pytorch_time,
                'triton_time_ms': triton_time,
                'speedup': speedup,
                'max_difference': max_diff,
                'is_correct': is_correct,
                'device': device
            }
            
            results[config['name']] = result
            
            print(f"✅ PyTorch: {pytorch_time:.2f}ms")
            print(f"⚡ Triton: {triton_time:.2f}ms")
            print(f"🚀 加速比: {speedup:.2f}x")
            print(f"🎯 数值误差: {max_diff:.2e} {'✅' if is_correct else '❌'}")
            
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            results[config['name']] = {'error': str(e)}
    
    return results

def benchmark_triton_pooling():
    """基准测试Triton池化内核"""
    print("\n🏊 测试 Triton 池化内核")
    print("-" * 40)
    
    try:
        from neuroexapt.kernels.pool_triton import avg_pool3x3_forward, max_pool3x3_forward, TRITON_AVAILABLE
        print(f"Triton可用: {TRITON_AVAILABLE}")
    except ImportError as e:
        print(f"❌ 无法导入Triton池化模块: {e}")
        return {}
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 测试配置
    test_configs = [
        {"name": "Small", "B": 4, "C": 32, "H": 32, "W": 32},
        {"name": "Medium", "B": 8, "C": 64, "H": 64, "W": 64},
        {"name": "Large", "B": 16, "C": 128, "H": 128, "W": 128},
    ]
    
    results = {'avgpool': {}, 'maxpool': {}}
    warmup_runs = 5
    benchmark_runs = 15
    
    for config in test_configs:
        print(f"\n🧪 测试配置: {config['name']}")
        B, C, H, W = config["B"], config["C"], config["H"], config["W"]
        
        x = torch.randn(B, C, H, W, device=device)
        print(f"输入形状: {list(x.shape)}")
        
        # 测试平均池化
        try:
            # 预热
            for _ in range(warmup_runs):
                _ = F.avg_pool2d(x, 3, stride=1, padding=1)
                _ = avg_pool3x3_forward(x, stride=1)
            
            # PyTorch平均池化
            if device == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            for _ in range(benchmark_runs):
                pytorch_avg = F.avg_pool2d(x, 3, stride=1, padding=1)
            if device == "cuda":
                torch.cuda.synchronize()
            pytorch_avg_time = (time.perf_counter() - start_time) / benchmark_runs * 1000
            
            # Triton平均池化
            if device == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            for _ in range(benchmark_runs):
                triton_avg = avg_pool3x3_forward(x, stride=1)
            if device == "cuda":
                torch.cuda.synchronize()
            triton_avg_time = (time.perf_counter() - start_time) / benchmark_runs * 1000
            
            # 验证平均池化正确性
            avg_diff = torch.max(torch.abs(pytorch_avg - triton_avg)).item()
            avg_correct = avg_diff < 1e-3
            avg_speedup = pytorch_avg_time / triton_avg_time if triton_avg_time > 0 else 0
            
            results['avgpool'][config['name']] = {
                'config': config,
                'pytorch_time_ms': pytorch_avg_time,
                'triton_time_ms': triton_avg_time,
                'speedup': avg_speedup,
                'max_difference': avg_diff,
                'is_correct': avg_correct,
                'device': device
            }
            
            print(f"平均池化:")
            print(f"  PyTorch: {pytorch_avg_time:.2f}ms")
            print(f"  Triton: {triton_avg_time:.2f}ms")
            print(f"  加速比: {avg_speedup:.2f}x")
            print(f"  误差: {avg_diff:.2e} {'✅' if avg_correct else '❌'}")
            
        except Exception as e:
            print(f"❌ 平均池化测试失败: {e}")
            results['avgpool'][config['name']] = {'error': str(e)}
        
        # 测试最大池化
        try:
            # 预热
            for _ in range(warmup_runs):
                _ = F.max_pool2d(x, 3, stride=1, padding=1)
                _ = max_pool3x3_forward(x, stride=1)
            
            # PyTorch最大池化
            if device == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            for _ in range(benchmark_runs):
                pytorch_max = F.max_pool2d(x, 3, stride=1, padding=1)
            if device == "cuda":
                torch.cuda.synchronize()
            pytorch_max_time = (time.perf_counter() - start_time) / benchmark_runs * 1000
            
            # Triton最大池化
            if device == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            for _ in range(benchmark_runs):
                triton_max = max_pool3x3_forward(x, stride=1)
            if device == "cuda":
                torch.cuda.synchronize()
            triton_max_time = (time.perf_counter() - start_time) / benchmark_runs * 1000
            
            # 验证最大池化正确性
            max_diff = torch.max(torch.abs(pytorch_max - triton_max)).item()
            max_correct = max_diff < 1e-3
            max_speedup = pytorch_max_time / triton_max_time if triton_max_time > 0 else 0
            
            results['maxpool'][config['name']] = {
                'config': config,
                'pytorch_time_ms': pytorch_max_time,
                'triton_time_ms': triton_max_time,
                'speedup': max_speedup,
                'max_difference': max_diff,
                'is_correct': max_correct,
                'device': device
            }
            
            print(f"最大池化:")
            print(f"  PyTorch: {pytorch_max_time:.2f}ms")
            print(f"  Triton: {triton_max_time:.2f}ms")
            print(f"  加速比: {max_speedup:.2f}x")
            print(f"  误差: {max_diff:.2e} {'✅' if max_correct else '❌'}")
            
        except Exception as e:
            print(f"❌ 最大池化测试失败: {e}")
            results['maxpool'][config['name']] = {'error': str(e)}
    
    return results

def generate_comprehensive_report(all_results):
    """生成综合报告"""
    print("\n" + "="*60)
    print("📊 NeuroExapt Triton 内核性能测试报告")
    print("="*60)
    
    # 统计所有成功的测试
    all_speedups = []
    operation_stats = {}
    
    # 处理SoftmaxSum结果
    if 'softmax_sum' in all_results:
        for name, result in all_results['softmax_sum'].items():
            if 'speedup' in result and result['speedup'] > 0:
                all_speedups.append(result['speedup'])
                if 'SoftmaxSum' not in operation_stats:
                    operation_stats['SoftmaxSum'] = []
                operation_stats['SoftmaxSum'].append(result['speedup'])
    
    # 处理分离卷积结果
    if 'sepconv' in all_results:
        for name, result in all_results['sepconv'].items():
            if 'speedup' in result and result['speedup'] > 0:
                all_speedups.append(result['speedup'])
                if 'SepConv' not in operation_stats:
                    operation_stats['SepConv'] = []
                operation_stats['SepConv'].append(result['speedup'])
    
    # 处理池化结果
    if 'pooling' in all_results:
        for pool_type in ['avgpool', 'maxpool']:
            if pool_type in all_results['pooling']:
                for name, result in all_results['pooling'][pool_type].items():
                    if 'speedup' in result and result['speedup'] > 0:
                        all_speedups.append(result['speedup'])
                        op_name = f"{'AvgPool' if pool_type == 'avgpool' else 'MaxPool'}"
                        if op_name not in operation_stats:
                            operation_stats[op_name] = []
                        operation_stats[op_name].append(result['speedup'])
    
    # 总体统计
    if all_speedups:
        print(f"🚀 总体性能统计:")
        print(f"   测试总数: {len(all_speedups)}")
        print(f"   平均加速比: {np.mean(all_speedups):.2f}x")
        print(f"   最大加速比: {np.max(all_speedups):.2f}x")
        print(f"   最小加速比: {np.min(all_speedups):.2f}x")
        print(f"   中位数加速比: {np.median(all_speedups):.2f}x")
        
        print(f"\n📈 按操作类型统计:")
        for op, speedups in operation_stats.items():
            avg_speedup = np.mean(speedups)
            max_speedup = np.max(speedups)
            print(f"   {op}: {avg_speedup:.2f}x 平均, {max_speedup:.2f}x 最大 ({len(speedups)} 个测试)")
        
        # 性能分级
        excellent = len([s for s in all_speedups if s > 2.0])
        good = len([s for s in all_speedups if 1.5 <= s <= 2.0])
        moderate = len([s for s in all_speedups if 1.0 <= s < 1.5])
        poor = len([s for s in all_speedups if s < 1.0])
        
        print(f"\n🏆 性能分级:")
        print(f"   优秀 (>2.0x): {excellent} 个测试")
        print(f"   良好 (1.5-2.0x): {good} 个测试")
        print(f"   一般 (1.0-1.5x): {moderate} 个测试")
        print(f"   较差 (<1.0x): {poor} 个测试")
        
        print(f"\n💡 建议:")
        if excellent > 0:
            print(f"   ✅ {excellent} 个测试显示显著性能提升，推荐在生产中使用")
        if good > 0:
            print(f"   ✅ {good} 个测试显示良好性能提升")
        if moderate > 0:
            print(f"   ⚠️  {moderate} 个测试显示适中性能提升，可根据具体场景选择")
        if poor > 0:
            print(f"   ❌ {poor} 个测试性能不佳，需要进一步优化")
    else:
        print("❌ 没有成功的性能测试结果")
    
    return {
        'total_tests': len(all_speedups),
        'avg_speedup': np.mean(all_speedups) if all_speedups else 0,
        'max_speedup': np.max(all_speedups) if all_speedups else 0,
        'operation_stats': operation_stats,
        'performance_grades': {
            'excellent': excellent if all_speedups else 0,
            'good': good if all_speedups else 0,
            'moderate': moderate if all_speedups else 0,
            'poor': poor if all_speedups else 0
        } if all_speedups else {}
    }

def save_results_to_file(all_results, summary):
    """保存结果到文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存详细结果
    os.makedirs("data/triton_benchmarks", exist_ok=True)
    detailed_file = f"data/triton_benchmarks/triton_kernels_benchmark_{timestamp}.json"
    
    with open(detailed_file, 'w', encoding='utf-8') as f:
        json.dump({
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'pytorch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'cuda_device': torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            },
            'detailed_results': all_results,
            'summary': summary
        }, f, indent=2, ensure_ascii=False)
    
    print(f"💾 详细结果已保存到: {detailed_file}")

def main():
    """主函数"""
    print("🧪 NeuroExapt Triton 内核基准测试")
    print("=" * 50)
    print("测试已实现的优化内核性能表现")
    print()
    
    all_results = {}
    
    # 运行各项测试
    print("🔥 第一项：CUDA SoftmaxSum 扩展测试")
    all_results['softmax_sum'] = benchmark_cuda_softmax_sum()
    
    print("\n⚡ 第二项：Triton 分离卷积内核测试")
    all_results['sepconv'] = benchmark_triton_sepconv()
    
    print("\n🏊 第三项：Triton 池化内核测试")
    all_results['pooling'] = benchmark_triton_pooling()
    
    # 生成综合报告
    summary = generate_comprehensive_report(all_results)
    
    # 保存结果
    save_results_to_file(all_results, summary)
    
    print(f"\n🎉 基准测试完成！")
    print(f"详细结果请查看 data/triton_benchmarks/ 目录")

if __name__ == "__main__":
    main() 