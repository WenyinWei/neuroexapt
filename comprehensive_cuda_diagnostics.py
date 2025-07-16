#!/usr/bin/env python3
"""
全面的CUDA环境诊断工具
用于排查PyTorch基础算子CUDA错误的根本原因
"""

import os
import sys
import torch
import subprocess
import gc
from pathlib import Path

def print_section(title):
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print(f"{'='*60}")

def print_subsection(title):
    print(f"\n{'─'*40}")
    print(f"📋 {title}")
    print(f"{'─'*40}")

def run_command(cmd, capture_output=True):
    """运行系统命令并返回结果"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=capture_output, text=True, timeout=30)
        return result.stdout.strip() if result.returncode == 0 else f"Error: {result.stderr.strip()}"
    except subprocess.TimeoutExpired:
        return "Error: Command timeout"
    except Exception as e:
        return f"Error: {str(e)}"

def check_system_info():
    """检查系统基础信息"""
    print_section("系统环境信息")
    
    print(f"操作系统: {os.name}")
    print(f"Python版本: {sys.version}")
    print(f"Python路径: {sys.executable}")
    
    # 检查WSL
    wsl_info = run_command("uname -r")
    if "microsoft" in wsl_info.lower():
        print(f"WSL环境: {wsl_info}")
        print("⚠️ 检测到WSL环境，可能存在GPU访问限制")
    else:
        print(f"内核版本: {wsl_info}")

def check_cuda_installation():
    """检查CUDA安装情况"""
    print_section("CUDA安装检查")
    
    # CUDA版本
    cuda_version = run_command("nvcc --version")
    print(f"NVCC版本:\n{cuda_version}")
    
    # NVIDIA驱动
    nvidia_smi = run_command("nvidia-smi")
    print(f"\nNVIDIA-SMI输出:\n{nvidia_smi}")
    
    # CUDA运行时库
    print(f"\n📦 CUDA库检查:")
    for lib in ["/usr/local/cuda/lib64/libcudart.so", "/usr/lib/x86_64-linux-gnu/libcuda.so"]:
        if Path(lib).exists():
            print(f"✅ {lib} - 存在")
        else:
            print(f"❌ {lib} - 不存在")

def check_pytorch_installation():
    """检查PyTorch安装情况"""
    print_section("PyTorch环境检查")
    
    print(f"PyTorch版本: {torch.__version__}")
    print(f"PyTorch路径: {torch.__file__}")
    
    # CUDA支持
    print(f"\n🔧 PyTorch CUDA支持:")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA设备数量: {torch.cuda.device_count()}")
        print(f"当前CUDA设备: {torch.cuda.current_device()}")
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"cuDNN版本: {torch.backends.cudnn.version()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\n📱 GPU {i}: {props.name}")
            print(f"   计算能力: {props.major}.{props.minor}")
            print(f"   总内存: {props.total_memory / 1024**3:.1f} GB")
            print(f"   多处理器数量: {props.multi_processor_count}")
    else:
        print("❌ CUDA不可用")

def check_gpu_memory():
    """检查GPU内存状态"""
    if not torch.cuda.is_available():
        print("⚠️ CUDA不可用，跳过GPU内存检查")
        return
        
    print_section("GPU内存状态")
    
    # 清理缓存
    torch.cuda.empty_cache()
    gc.collect()
    
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        total_memory = torch.cuda.get_device_properties(i).total_memory
        allocated = torch.cuda.memory_allocated(i)
        cached = torch.cuda.memory_reserved(i)
        
        print(f"\n📱 GPU {i}:")
        print(f"   总内存: {total_memory / 1024**3:.2f} GB")
        print(f"   已分配: {allocated / 1024**3:.2f} GB ({allocated/total_memory*100:.1f}%)")
        print(f"   已缓存: {cached / 1024**3:.2f} GB ({cached/total_memory*100:.1f}%)")
        print(f"   可用内存: {(total_memory - allocated) / 1024**3:.2f} GB")

def test_basic_cuda_operations():
    """测试基础CUDA操作"""
    print_section("基础CUDA操作测试")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，无法进行CUDA操作测试")
        return False
    
    try:
        print("🔸 测试1: 创建CUDA张量...")
        x = torch.randn(10, device='cuda')
        print(f"✅ 成功创建CUDA张量: {x.shape}")
        
        print("🔸 测试2: 基础数学运算...")
        y = x + 1
        print(f"✅ 加法运算成功: {y.mean().item():.4f}")
        
        print("🔸 测试3: 矩阵乘法...")
        A = torch.randn(100, 100, device='cuda')
        B = torch.randn(100, 100, device='cuda')
        C = torch.mm(A, B)
        print(f"✅ 矩阵乘法成功: {C.shape}")
        
        print("🔸 测试4: GPU-CPU数据传输...")
        cpu_data = C.cpu()
        gpu_data = cpu_data.cuda()
        print(f"✅ 数据传输成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 基础CUDA操作失败: {e}")
        return False

def test_pytorch_basic_ops():
    """测试PyTorch基础算子"""
    print_section("PyTorch基础算子测试")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，使用CPU测试")
        device = 'cpu'
    else:
        device = 'cuda'
    
    tests = [
        ("ReLU激活", lambda: torch.relu(torch.randn(10, 10, device=device))),
        ("Conv2D卷积", lambda: torch.nn.functional.conv2d(
            torch.randn(1, 3, 32, 32, device=device),
            torch.randn(16, 3, 3, 3, device=device)
        )),
        ("BatchNorm2D", lambda: torch.nn.functional.batch_norm(
            torch.randn(1, 16, 32, 32, device=device),
            torch.zeros(16, device=device),
            torch.ones(16, device=device),
            torch.zeros(16, device=device),
            torch.ones(16, device=device)
        )),
        ("MaxPool2D", lambda: torch.nn.functional.max_pool2d(
            torch.randn(1, 16, 32, 32, device=device), 3, stride=1, padding=1
        )),
        ("AdaptiveAvgPool2D", lambda: torch.nn.functional.adaptive_avg_pool2d(
            torch.randn(1, 16, 32, 32, device=device), (1, 1)
        ))
    ]
    
    success_count = 0
    for test_name, test_func in tests:
        try:
            print(f"🔸 测试: {test_name}...")
            result = test_func()
            print(f"✅ {test_name} 成功: {result.shape}")
            success_count += 1
        except Exception as e:
            print(f"❌ {test_name} 失败: {e}")
    
    print(f"\n📊 基础算子测试结果: {success_count}/{len(tests)} 通过")
    return success_count == len(tests)

def test_minimal_network():
    """测试最小神经网络"""
    print_section("最小神经网络测试")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，使用CPU测试")
        device = 'cpu'
    else:
        device = 'cuda'
    
    try:
        print("🔸 创建最小网络...")
        
        class MinimalNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 16, 3, padding=1, bias=False)
                self.bn = torch.nn.BatchNorm2d(16)
                self.relu = torch.nn.ReLU()
                self.pool = torch.nn.AdaptiveAvgPool2d(1)
                self.fc = torch.nn.Linear(16, 10)
            
            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                x = self.relu(x)
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x
        
        model = MinimalNet().to(device)
        print(f"✅ 网络创建成功")
        
        print("🔸 前向传播测试...")
        input_data = torch.randn(2, 3, 32, 32, device=device)
        output = model(input_data)
        print(f"✅ 前向传播成功: {output.shape}")
        
        print("🔸 反向传播测试...")
        loss = output.sum()
        loss.backward()
        print(f"✅ 反向传播成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 最小网络测试失败: {e}")
        return False

def test_with_cuda_dsa():
    """使用CUDA DSA重新测试失败的操作"""
    print_section("CUDA DSA诊断测试")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，无法进行DSA测试")
        return
    
    # 启用CUDA DSA
    os.environ['TORCH_USE_CUDA_DSA'] = '1'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    print("🔧 已启用 TORCH_USE_CUDA_DSA 和 CUDA_LAUNCH_BLOCKING")
    print("🔸 重新测试ReLU操作...")
    
    try:
        # 测试最简单的ReLU操作
        x = torch.randn(10, 10, device='cuda')
        y = torch.relu(x)
        print("✅ ReLU操作成功")
        
        # 测试Conv2D操作
        print("🔸 重新测试Conv2D操作...")
        input_tensor = torch.randn(1, 3, 32, 32, device='cuda')
        weight = torch.randn(16, 3, 3, 3, device='cuda')
        output = torch.nn.functional.conv2d(input_tensor, weight)
        print("✅ Conv2D操作成功")
        
    except Exception as e:
        print(f"❌ DSA诊断中发现错误: {e}")
        print("这可能提供了更详细的错误信息")

def check_environment_variables():
    """检查相关环境变量"""
    print_section("环境变量检查")
    
    important_vars = [
        'CUDA_VISIBLE_DEVICES',
        'CUDA_DEVICE_ORDER', 
        'TORCH_USE_CUDA_DSA',
        'CUDA_LAUNCH_BLOCKING',
        'PYTHONPATH',
        'LD_LIBRARY_PATH',
        'CUDA_HOME',
        'CUDA_PATH'
    ]
    
    for var in important_vars:
        value = os.environ.get(var)
        if value:
            print(f"✅ {var} = {value}")
        else:
            print(f"⚪ {var} = 未设置")

def generate_recommendations():
    """生成修复建议"""
    print_section("修复建议")
    
    print("基于诊断结果，建议按以下顺序尝试修复:")
    print()
    print("🔧 立即尝试的修复方案:")
    print("1. 重启Python进程和清理GPU内存")
    print("2. 设置环境变量后重试:")
    print("   export TORCH_USE_CUDA_DSA=1")
    print("   export CUDA_LAUNCH_BLOCKING=1")
    print("3. 降低batch size到最小(如16或32)")
    print("4. 使用CPU模式验证代码逻辑")
    print()
    print("🔧 中级修复方案:")
    print("1. 重新安装PyTorch:")
    print("   pip uninstall torch torchvision")
    print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    print("2. 更新NVIDIA驱动")
    print("3. 检查WSL2的GPU支持配置")
    print()
    print("🔧 高级修复方案:")
    print("1. 重装CUDA Toolkit")
    print("2. 检查GPU硬件状态")
    print("3. 尝试使用Docker容器隔离环境")

def main():
    """主诊断流程"""
    print("🏥 CUDA环境全面诊断工具")
    print("=" * 60)
    
    # 基础信息检查
    check_system_info()
    check_environment_variables()
    check_cuda_installation()
    check_pytorch_installation()
    
    # GPU和内存检查
    check_gpu_memory()
    
    # 功能测试
    cuda_basic_ok = test_basic_cuda_operations()
    pytorch_ops_ok = test_pytorch_basic_ops()
    network_ok = test_minimal_network()
    
    # 如果基础操作失败，尝试DSA诊断
    if not (cuda_basic_ok and pytorch_ops_ok):
        test_with_cuda_dsa()
    
    # 生成建议
    generate_recommendations()
    
    # 总结
    print_section("诊断总结")
    print(f"CUDA基础操作: {'✅ 正常' if cuda_basic_ok else '❌ 异常'}")
    print(f"PyTorch算子: {'✅ 正常' if pytorch_ops_ok else '❌ 异常'}")
    print(f"神经网络: {'✅ 正常' if network_ok else '❌ 异常'}")
    
    if cuda_basic_ok and pytorch_ops_ok and network_ok:
        print("\n🎉 所有测试通过！问题可能在于特定的模型配置或数据")
        print("建议检查:")
        print("- 模型复杂度和内存使用")
        print("- 特定的网络架构组件")
        print("- 数据加载和预处理步骤")
    else:
        print("\n⚠️ 发现环境问题，需要修复CUDA环境后再运行NeuroExapt")

if __name__ == "__main__":
    main() 