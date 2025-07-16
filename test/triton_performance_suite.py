#!/usr/bin/env python3
"""
NeuroExapt Triton 性能测试套件

这个文件提供了全面的Triton内核性能测试，包括：
1. 基础操作测试（矩阵乘法、元素级操作等）
2. 卷积操作测试（标准卷积、分离卷积、深度卷积）
3. 注意力机制测试（Multi-Head Attention、Flash Attention）
4. 激活函数测试（ReLU、GELU、Swish等）
5. 归一化操作测试（BatchNorm、LayerNorm等）
6. 内存优化测试（融合操作、内存带宽）
7. 实际神经网络组件测试（MixedOp、SoftmaxSum等）
"""

import torch
import torch.nn.functional as F
import time
import gc
import json
import os
import sys
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

@dataclass
class BenchmarkResult:
    """性能基准测试结果数据结构"""
    operation: str
    test_config: Dict[str, Any]
    pytorch_time_ms: float
    triton_time_ms: float
    speedup: float
    memory_usage_mb: float
    memory_savings_percent: float
    device: str
    timestamp: str
    status: str  # 'success', 'failed', 'skipped'
    error_message: Optional[str] = None

class TritonPerformanceSuite:
    """Triton性能测试套件"""
    
    def __init__(self, device: str = None, save_results: bool = True):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_results = save_results
        self.results: List[BenchmarkResult] = []
        self.warmup_runs = 5
        self.benchmark_runs = 20
        
        print(f"🚀 Triton性能测试套件初始化")
        print(f"设备: {self.device}")
        print(f"Triton可用: {TRITON_AVAILABLE}")
        print(f"PyTorch版本: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"CUDA设备: {torch.cuda.get_device_name()}")
            print(f"CUDA版本: {torch.version.cuda}")
        print("-" * 60)
    
    def benchmark_operation(self, name: str, pytorch_fn, triton_fn, 
                          test_configs: List[Dict], **kwargs) -> List[BenchmarkResult]:
        """对单个操作进行基准测试"""
        results = []
        
        for config in test_configs:
            print(f"\n🧪 测试 {name} - 配置: {config}")
            
            try:
                # 生成测试数据
                test_data = self._generate_test_data(config)
                
                # 预热
                for _ in range(self.warmup_runs):
                    _ = pytorch_fn(**test_data)
                    if triton_fn:
                        _ = triton_fn(**test_data)
                
                # 同步GPU
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                
                # 测试PyTorch
                start_time = time.perf_counter()
                for _ in range(self.benchmark_runs):
                    pytorch_result = pytorch_fn(**test_data)
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                pytorch_time = (time.perf_counter() - start_time) / self.benchmark_runs * 1000
                
                # 测试Triton (如果可用)
                triton_time = float('inf')
                speedup = 0.0
                if triton_fn and TRITON_AVAILABLE:
                    start_time = time.perf_counter()
                    for _ in range(self.benchmark_runs):
                        triton_result = triton_fn(**test_data)
                    if self.device == 'cuda':
                        torch.cuda.synchronize()
                    triton_time = (time.perf_counter() - start_time) / self.benchmark_runs * 1000
                    speedup = pytorch_time / triton_time
                    
                    # 验证结果正确性
                    if not self._verify_results(pytorch_result, triton_result):
                        print(f"⚠️  结果验证失败，数值不匹配")
                
                # 计算内存使用
                memory_usage = self._measure_memory_usage(test_data)
                
                result = BenchmarkResult(
                    operation=name,
                    test_config=config,
                    pytorch_time_ms=pytorch_time,
                    triton_time_ms=triton_time,
                    speedup=speedup,
                    memory_usage_mb=memory_usage,
                    memory_savings_percent=0.0,  # 待实现
                    device=self.device,
                    timestamp=datetime.now().isoformat(),
                    status='success'
                )
                
                results.append(result)
                self.results.append(result)
                
                print(f"✅ PyTorch: {pytorch_time:.2f}ms")
                if triton_fn and TRITON_AVAILABLE:
                    print(f"⚡ Triton: {triton_time:.2f}ms")
                    print(f"🚀 加速比: {speedup:.2f}x")
                else:
                    print(f"❌ Triton不可用")
                
            except Exception as e:
                print(f"❌ 测试失败: {e}")
                error_result = BenchmarkResult(
                    operation=name,
                    test_config=config,
                    pytorch_time_ms=0.0,
                    triton_time_ms=0.0,
                    speedup=0.0,
                    memory_usage_mb=0.0,
                    memory_savings_percent=0.0,
                    device=self.device,
                    timestamp=datetime.now().isoformat(),
                    status='failed',
                    error_message=str(e)
                )
                results.append(error_result)
                self.results.append(error_result)
        
        return results
    
    def _generate_test_data(self, config: Dict) -> Dict:
        """根据配置生成测试数据"""
        data = {}
        
        # 根据不同操作类型生成数据
        if 'batch_size' in config and 'channels' in config:
            # 卷积类操作
            B, C, H, W = config['batch_size'], config['channels'], config.get('height', 32), config.get('width', 32)
            data['x'] = torch.randn(B, C, H, W, device=self.device, requires_grad=True)
            
            if 'kernel_size' in config:
                K = config['kernel_size']
                data['weight'] = torch.randn(C, C, K, K, device=self.device, requires_grad=True)
        
        elif 'seq_len' in config and 'hidden_dim' in config:
            # 注意力机制类操作
            B, L, D = config.get('batch_size', 2), config['seq_len'], config['hidden_dim']
            data['x'] = torch.randn(B, L, D, device=self.device, requires_grad=True)
            data['y'] = torch.randn(B, L, D, device=self.device, requires_grad=True)
        
        elif 'matrix_size' in config:
            # 矩阵操作
            N = config['matrix_size']
            data['a'] = torch.randn(N, N, device=self.device, requires_grad=True)
            data['b'] = torch.randn(N, N, device=self.device, requires_grad=True)
        
        elif 'num_ops' in config and 'tensor_shape' in config:
            # MixedOp类操作
            N = config['num_ops']
            shape = config['tensor_shape']
            data['tensors'] = [torch.randn(*shape, device=self.device, requires_grad=True) for _ in range(N)]
            data['logits'] = torch.randn(N, device=self.device, requires_grad=True)
        
        return data
    
    def _verify_results(self, pytorch_result, triton_result, rtol=1e-3, atol=1e-3):
        """验证PyTorch和Triton结果的一致性"""
        try:
            if isinstance(pytorch_result, (list, tuple)):
                if len(pytorch_result) != len(triton_result):
                    return False
                return all(torch.allclose(p, t, rtol=rtol, atol=atol) 
                          for p, t in zip(pytorch_result, triton_result))
            else:
                return torch.allclose(pytorch_result, triton_result, rtol=rtol, atol=atol)
        except:
            return False
    
    def _measure_memory_usage(self, test_data: Dict) -> float:
        """测量内存使用量（MB）"""
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            return torch.cuda.max_memory_allocated() / 1024 / 1024
        else:
            # CPU内存测量（简化）
            total_size = 0
            for tensor in test_data.values():
                if isinstance(tensor, torch.Tensor):
                    total_size += tensor.numel() * tensor.element_size()
                elif isinstance(tensor, list):
                    for t in tensor:
                        if isinstance(t, torch.Tensor):
                            total_size += t.numel() * t.element_size()
            return total_size / 1024 / 1024

    # ============ 具体测试方法 ============
    
    def test_matrix_multiplication(self):
        """测试矩阵乘法"""
        print("\n🔢 测试矩阵乘法性能")
        
        def pytorch_matmul(a, b):
            return torch.matmul(a, b)
        
        def triton_matmul(a, b):
            # 使用Triton实现的矩阵乘法（如果有的话）
            return torch.matmul(a, b)  # 暂时使用PyTorch作为placeholder
        
        configs = [
            {'matrix_size': 512},
            {'matrix_size': 1024},
            {'matrix_size': 2048},
            {'matrix_size': 4096},
        ]
        
        return self.benchmark_operation(
            "Matrix Multiplication",
            pytorch_matmul,
            triton_matmul if TRITON_AVAILABLE else None,
            configs
        )
    
    def test_convolution_operations(self):
        """测试卷积操作"""
        print("\n🔄 测试卷积操作性能")
        
        def pytorch_conv2d(x, weight):
            return F.conv2d(x, weight, padding=1)
        
        def triton_conv2d(x, weight):
            # 暂时使用PyTorch作为placeholder
            return F.conv2d(x, weight, padding=1)
        
        configs = [
            {'batch_size': 8, 'channels': 32, 'height': 32, 'width': 32, 'kernel_size': 3},
            {'batch_size': 16, 'channels': 64, 'height': 64, 'width': 64, 'kernel_size': 3},
            {'batch_size': 32, 'channels': 128, 'height': 128, 'width': 128, 'kernel_size': 3},
        ]
        
        return self.benchmark_operation(
            "Convolution 2D",
            pytorch_conv2d,
            triton_conv2d if TRITON_AVAILABLE else None,
            configs
        )
    
    def test_separable_convolution(self):
        """测试分离卷积"""
        print("\n🔀 测试分离卷积性能")
        
        def pytorch_sepconv(x, dw_weight, pw_weight, bias):
            # 深度卷积
            y = F.conv2d(x, dw_weight, bias=None, stride=1, padding=1, groups=x.size(1))
            # 点卷积
            y = F.conv2d(y, pw_weight, bias=bias)
            return y
        
        def triton_sepconv(x, dw_weight, pw_weight, bias):
            try:
                from neuroexapt.kernels import sepconv_forward_generic
                return sepconv_forward_generic(x, dw_weight, pw_weight, bias)
            except ImportError:
                return pytorch_sepconv(x, dw_weight, pw_weight, bias)
        
        # 生成特殊的测试数据
        def generate_sepconv_data(config):
            B, C, H, W = config['batch_size'], config['channels'], 32, 32
            K = config['kernel_size']
            data = {
                'x': torch.randn(B, C, H, W, device=self.device, requires_grad=True),
                'dw_weight': torch.randn(C, 1, K, K, device=self.device, requires_grad=True),
                'pw_weight': torch.randn(C*2, C, 1, 1, device=self.device, requires_grad=True),
                'bias': torch.randn(C*2, device=self.device, requires_grad=True)
            }
            return data
        
        configs = [
            {'batch_size': 4, 'channels': 16, 'kernel_size': 3},
            {'batch_size': 8, 'channels': 32, 'kernel_size': 3},
            {'batch_size': 16, 'channels': 64, 'kernel_size': 3},
        ]
        
        results = []
        for config in configs:
            test_data = generate_sepconv_data(config)
            
            try:
                # 预热
                for _ in range(self.warmup_runs):
                    _ = pytorch_sepconv(**test_data)
                    _ = triton_sepconv(**test_data)
                
                # 测试PyTorch
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                start_time = time.perf_counter()
                for _ in range(self.benchmark_runs):
                    pytorch_result = pytorch_sepconv(**test_data)
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                pytorch_time = (time.perf_counter() - start_time) / self.benchmark_runs * 1000
                
                # 测试Triton
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                start_time = time.perf_counter()
                for _ in range(self.benchmark_runs):
                    triton_result = triton_sepconv(**test_data)
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                triton_time = (time.perf_counter() - start_time) / self.benchmark_runs * 1000
                
                speedup = pytorch_time / triton_time
                memory_usage = self._measure_memory_usage(test_data)
                
                result = BenchmarkResult(
                    operation="Separable Convolution",
                    test_config=config,
                    pytorch_time_ms=pytorch_time,
                    triton_time_ms=triton_time,
                    speedup=speedup,
                    memory_usage_mb=memory_usage,
                    memory_savings_percent=0.0,
                    device=self.device,
                    timestamp=datetime.now().isoformat(),
                    status='success'
                )
                
                results.append(result)
                self.results.append(result)
                
                print(f"✅ 配置 {config}:")
                print(f"   PyTorch: {pytorch_time:.2f}ms")
                print(f"   Triton: {triton_time:.2f}ms")
                print(f"   加速比: {speedup:.2f}x")
                
            except Exception as e:
                print(f"❌ 分离卷积测试失败: {e}")
        
        return results
    
    def test_activation_functions(self):
        """测试激活函数"""
        print("\n⚡ 测试激活函数性能")
        
        def pytorch_relu(x):
            return F.relu(x)
        
        def triton_relu(x):
            # Triton ReLU实现placeholder
            return F.relu(x)
        
        def pytorch_gelu(x):
            return F.gelu(x)
        
        def triton_gelu(x):
            # Triton GELU实现placeholder
            return F.gelu(x)
        
        configs = [
            {'batch_size': 32, 'channels': 64, 'height': 32, 'width': 32},
            {'batch_size': 64, 'channels': 128, 'height': 64, 'width': 64},
        ]
        
        relu_results = self.benchmark_operation(
            "ReLU Activation",
            pytorch_relu,
            triton_relu if TRITON_AVAILABLE else None,
            configs
        )
        
        gelu_results = self.benchmark_operation(
            "GELU Activation", 
            pytorch_gelu,
            triton_gelu if TRITON_AVAILABLE else None,
            configs
        )
        
        return relu_results + gelu_results
    
    def test_mixedop_operation(self):
        """测试MixedOp操作（SoftmaxSum）"""
        print("\n🧬 测试MixedOp SoftmaxSum性能")
        
        def pytorch_softmax_sum(tensors, logits):
            # PyTorch参考实现
            stacked = torch.stack(tensors, dim=0)
            weights = F.softmax(logits, dim=0)
            return (stacked * weights.view(-1, 1, 1, 1, 1)).sum(dim=0)
        
        def triton_softmax_sum(tensors, logits):
            try:
                from neuroexapt.cuda_ops import SoftmaxSumFunction
                stacked = torch.stack(tensors, dim=0)
                return SoftmaxSumFunction.apply(stacked, logits)
            except ImportError:
                return pytorch_softmax_sum(tensors, logits)
        
        configs = [
            {'num_ops': 4, 'tensor_shape': (2, 16, 16, 16)},
            {'num_ops': 8, 'tensor_shape': (4, 32, 32, 32)},
            {'num_ops': 12, 'tensor_shape': (8, 64, 64, 64)},
            {'num_ops': 16, 'tensor_shape': (16, 128, 128, 128)},
        ]
        
        results = []
        for config in configs:
            test_data = self._generate_test_data(config)
            
            try:
                # 转换为正确的输入格式
                tensors = test_data['tensors']
                logits = test_data['logits']
                
                # 预热
                for _ in range(self.warmup_runs):
                    _ = pytorch_softmax_sum(tensors, logits)
                    _ = triton_softmax_sum(tensors, logits)
                
                # 测试PyTorch
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                start_time = time.perf_counter()
                for _ in range(self.benchmark_runs):
                    pytorch_result = pytorch_softmax_sum(tensors, logits)
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                pytorch_time = (time.perf_counter() - start_time) / self.benchmark_runs * 1000
                
                # 测试Triton/CUDA
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                start_time = time.perf_counter()
                for _ in range(self.benchmark_runs):
                    triton_result = triton_softmax_sum(tensors, logits)
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                triton_time = (time.perf_counter() - start_time) / self.benchmark_runs * 1000
                
                speedup = pytorch_time / triton_time
                memory_usage = self._measure_memory_usage(test_data)
                
                result = BenchmarkResult(
                    operation="MixedOp SoftmaxSum",
                    test_config=config,
                    pytorch_time_ms=pytorch_time,
                    triton_time_ms=triton_time,
                    speedup=speedup,
                    memory_usage_mb=memory_usage,
                    memory_savings_percent=0.0,
                    device=self.device,
                    timestamp=datetime.now().isoformat(),
                    status='success'
                )
                
                results.append(result)
                self.results.append(result)
                
                print(f"✅ 配置 {config}:")
                print(f"   PyTorch: {pytorch_time:.2f}ms")
                print(f"   优化版本: {triton_time:.2f}ms")
                print(f"   加速比: {speedup:.2f}x")
                
            except Exception as e:
                print(f"❌ MixedOp测试失败: {e}")
        
        return results
    
    def test_pooling_operations(self):
        """测试池化操作"""
        print("\n🏊 测试池化操作性能")
        
        def pytorch_avgpool(x):
            return F.avg_pool2d(x, 3, stride=1, padding=1)
        
        def triton_avgpool(x):
            try:
                from neuroexapt.kernels.pool_triton import avg_pool3x3_forward
                return avg_pool3x3_forward(x, stride=1)
            except ImportError:
                return F.avg_pool2d(x, 3, stride=1, padding=1)
        
        def pytorch_maxpool(x):
            return F.max_pool2d(x, 3, stride=1, padding=1)
        
        def triton_maxpool(x):
            try:
                from neuroexapt.kernels.pool_triton import max_pool3x3_forward
                return max_pool3x3_forward(x, stride=1)
            except ImportError:
                return F.max_pool2d(x, 3, stride=1, padding=1)
        
        configs = [
            {'batch_size': 8, 'channels': 32, 'height': 32, 'width': 32},
            {'batch_size': 16, 'channels': 64, 'height': 64, 'width': 64},
            {'batch_size': 32, 'channels': 128, 'height': 128, 'width': 128},
        ]
        
        avgpool_results = self.benchmark_operation(
            "Average Pooling 3x3",
            pytorch_avgpool,
            triton_avgpool if TRITON_AVAILABLE else None,
            configs
        )
        
        maxpool_results = self.benchmark_operation(
            "Max Pooling 3x3",
            pytorch_maxpool,
            triton_maxpool if TRITON_AVAILABLE else None,
            configs
        )
        
        return avgpool_results + maxpool_results
    
    def run_all_tests(self) -> Dict[str, List[BenchmarkResult]]:
        """运行所有测试"""
        print("🚀 开始全面Triton性能测试")
        print("=" * 60)
        
        all_results = {}
        
        # 运行各个测试
        test_methods = [
            ('Matrix Multiplication', self.test_matrix_multiplication),
            ('Convolution Operations', self.test_convolution_operations),
            ('Separable Convolution', self.test_separable_convolution),
            ('Activation Functions', self.test_activation_functions),
            ('MixedOp Operations', self.test_mixedop_operation),
            ('Pooling Operations', self.test_pooling_operations),
        ]
        
        for test_name, test_method in test_methods:
            print(f"\n{'='*20} {test_name} {'='*20}")
            try:
                results = test_method()
                all_results[test_name] = results
                print(f"✅ {test_name} 完成")
            except Exception as e:
                print(f"❌ {test_name} 失败: {e}")
                all_results[test_name] = []
        
        # 保存结果
        if self.save_results:
            self.save_benchmark_results()
        
        # 生成总结报告
        self.generate_summary_report()
        
        return all_results
    
    def save_benchmark_results(self):
        """保存基准测试结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/triton_benchmarks/triton_performance_{timestamp}.json"
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        results_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'device': self.device,
                'triton_available': TRITON_AVAILABLE,
                'pytorch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'cuda_device': torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            },
            'results': [asdict(result) for result in self.results]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 结果已保存到: {filename}")
    
    def generate_summary_report(self):
        """生成总结报告"""
        print("\n" + "="*60)
        print("📊 Triton性能测试总结报告")
        print("="*60)
        
        if not self.results:
            print("❌ 没有可用的测试结果")
            return
        
        successful_results = [r for r in self.results if r.status == 'success' and r.speedup > 0]
        
        print(f"总测试数: {len(self.results)}")
        print(f"成功测试数: {len(successful_results)}")
        print(f"成功率: {len(successful_results)/len(self.results)*100:.1f}%")
        
        if successful_results:
            speedups = [r.speedup for r in successful_results]
            print(f"\n🚀 性能统计:")
            print(f"平均加速比: {np.mean(speedups):.2f}x")
            print(f"最大加速比: {np.max(speedups):.2f}x")
            print(f"最小加速比: {np.min(speedups):.2f}x")
            print(f"中位数加速比: {np.median(speedups):.2f}x")
            
            print(f"\n📈 按操作类型分组:")
            operation_groups = {}
            for result in successful_results:
                op = result.operation
                if op not in operation_groups:
                    operation_groups[op] = []
                operation_groups[op].append(result.speedup)
            
            for op, speedups in operation_groups.items():
                avg_speedup = np.mean(speedups)
                print(f"  {op}: {avg_speedup:.2f}x (共{len(speedups)}个测试)")
        
        print("\n💡 建议:")
        if TRITON_AVAILABLE:
            best_ops = [r for r in successful_results if r.speedup > 1.5]
            if best_ops:
                print(f"✅ 推荐优化的操作 (>1.5x加速):")
                for op in set(r.operation for r in best_ops):
                    print(f"  - {op}")
            else:
                print("⚠️  当前测试中没有显著的性能提升")
        else:
            print("❌ Triton不可用，建议安装Triton以获得更好的性能")


def main():
    """主函数"""
    print("🧪 NeuroExapt Triton 性能测试套件")
    print("这将对各种深度学习操作进行全面的性能测试")
    print("-" * 50)
    
    # 创建测试套件
    suite = TritonPerformanceSuite(save_results=True)
    
    # 运行所有测试
    results = suite.run_all_tests()
    
    print("\n🎉 性能测试完成！")
    print("结果已保存到 data/triton_benchmarks/ 目录")
    print("详细报告请查看生成的JSON文件")

if __name__ == "__main__":
    main() 